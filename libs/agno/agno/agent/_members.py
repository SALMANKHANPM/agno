"""Member coordination helpers for Agent."""

from __future__ import annotations

import asyncio
import contextlib
import json
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, AsyncIterator, Dict, Iterator, List, Optional, Sequence, Tuple, Union

from pydantic import BaseModel

from agno.agent import Agent
from agno.media import Audio, File, Image, Video
from agno.models.message import Message
from agno.run import RunContext
from agno.run.agent import RunOutput, RunOutputEvent
from agno.run.team import TeamRunOutput, TeamRunOutputEvent
from agno.session import AgentSession
from agno.tools import Toolkit
from agno.tools.function import Function
from agno.utils.log import log_info, use_agent_logger
from agno.utils.merge_dict import merge_dictionaries
from agno.utils.response import check_if_run_cancelled
from agno.utils.team import get_member_id

if TYPE_CHECKING:
    from agno.team.team import Team


Member = Union[Agent, "Team"]


def _get_tool_names(member: Any, async_mode: bool = False) -> List[str]:
    tool_names: List[str] = []
    if member.tools is None or not isinstance(member.tools, list):
        return tool_names
    for tool in member.tools:
        if isinstance(tool, Toolkit):
            toolkit_functions = tool.get_async_functions() if async_mode else tool.get_functions()
            tool_names.extend(func.name for func in toolkit_functions.values() if func.entrypoint)
        elif isinstance(tool, Function) and tool.entrypoint:
            tool_names.append(tool.name)
        elif callable(tool):
            tool_names.append(tool.__name__)
        elif isinstance(tool, dict) and tool.get("name") is not None:
            tool_names.append(str(tool["name"]))
        else:
            tool_names.append(str(tool))
    return tool_names


def get_members_system_message_content(
    agent: Agent, indent: int = 0, run_context: Optional[RunContext] = None, async_mode: bool = False
) -> str:
    from agno.team.team import Team
    from agno.utils.callables import get_resolved_members

    pad = " " * indent
    content = ""
    resolved_members = get_resolved_members(agent, run_context)
    if not resolved_members:
        return content

    for member in resolved_members:
        member_id = get_member_id(member)
        if isinstance(member, Team):
            content += f'{pad}<member id="{member_id}" name="{member.name}" type="team">\n'
            if member.description is not None:
                content += f"{pad}  Description: {member.description}\n"
            if member.members is not None:
                content += member.get_members_system_message_content(
                    indent=indent + 2, run_context=run_context, async_mode=async_mode
                )
            content += f"{pad}</member>\n"
        else:
            content += f'{pad}<member id="{member_id}" name="{member.name}">\n'
            if member.role is not None:
                content += f"{pad}  Role: {member.role}\n"
            if member.description is not None:
                content += f"{pad}  Description: {member.description}\n"
            if agent.add_member_tools_to_context:
                tool_names = _get_tool_names(member, async_mode=async_mode)
                if tool_names:
                    content += f"{pad}  Tools: {', '.join(tool_names)}\n"
            content += f"{pad}</member>\n"

    return content


def build_members_context(agent: Agent, run_context: Optional[RunContext] = None, async_mode: bool = False) -> str:
    from agno.utils.callables import get_resolved_members

    resolved_members = get_resolved_members(agent, run_context)
    if not resolved_members:
        return ""

    content = (
        "You can coordinate specialized member agents to fulfill the user's request. "
        "Delegate to members when their expertise or tools are needed. "
        "For straightforward requests you can handle directly, respond without delegating.\n\n"
        "<members>\n"
    )
    content += get_members_system_message_content(agent, run_context=run_context, async_mode=async_mode)
    content += "</members>\n\n"
    content += (
        "<how_to_delegate>\n"
        "- Match each sub-task to the member whose role and tools are the best fit.\n"
        "- Use only the member's ID when delegating.\n"
        "- Write self-contained task descriptions with the goal, relevant context, and expected output.\n"
        "- After member responses arrive, synthesize them into one coherent answer.\n"
        "</how_to_delegate>\n\n"
    )
    return content


def initialize_member(agent: Agent, member: Member, debug_mode: Optional[bool] = None) -> None:
    from agno.team.team import Team

    if debug_mode:
        member.debug_mode = True
        member.debug_level = agent.debug_level

    if isinstance(member, Agent):
        member.team_id = agent.id
        member._team = agent
        member.set_id()
        if member.model is None and agent.model is not None:
            member.model = agent.model
            log_info(f"Agent '{member.name or member.id}' inheriting model from Agent: {agent.model.id}")
    elif isinstance(member, Team):
        member.parent_team_id = agent.id
        member.set_id()
        member._set_default_model()
        if isinstance(member.members, list):
            for sub_member in member.members:
                member._initialize_member(sub_member, debug_mode=debug_mode)


def find_member_by_id(
    agent: Agent, member_id: str, run_context: Optional[RunContext] = None
) -> Optional[Tuple[int, Member]]:
    from agno.team.team import Team
    from agno.utils.callables import get_resolved_members

    resolved_members = get_resolved_members(agent, run_context)
    if resolved_members is None:
        return None

    for index, member in enumerate(resolved_members):
        if get_member_id(member) == member_id:
            return index, member
        if isinstance(member, Team):
            result = member._find_member_by_id(member_id, run_context=run_context)
            if result is not None:
                return result

    return None


def _get_history_for_member(session: AgentSession, member: Member) -> List[Message]:
    member_agent_id = member.id if isinstance(member, Agent) else None
    if member_agent_id is None:
        return []

    member_runs = [
        run
        for run in session.runs or []
        if isinstance(run, RunOutput) and run.agent_id == member_agent_id and not run.is_paused
    ]
    if member.num_history_runs is not None:
        member_runs = member_runs[-member.num_history_runs :]

    history: List[Message] = []
    for run in member_runs:
        for message in run.messages or []:
            if message.role in ["system", "tool"]:
                continue
            if message.from_history:
                continue
            history.append(message)
    if member.num_history_messages is not None:
        history = history[-member.num_history_messages :]

    history_copy = [deepcopy(msg) for msg in history]
    for message in history_copy:
        message.from_history = True
    return history_copy


def _stringify_member_response(run_response: Optional[Union[RunOutput, TeamRunOutput]], member_name: str) -> str:
    if run_response is None:
        return f"Agent {member_name}: No response from the member agent."
    if run_response.is_paused:
        return f"Agent {member_name}: Requires human input before continuing."
    try:
        if run_response.content is None and (run_response.tools is None or len(run_response.tools) == 0):
            return f"Agent {member_name}: No response from the member agent."
        if isinstance(run_response.content, str):
            if run_response.content.strip():
                return f"Agent {member_name}: {run_response.content}"
            if run_response.tools:
                return (
                    f"Agent {member_name}: {','.join([str(tool.result) for tool in run_response.tools if tool.result])}"
                )
        if issubclass(type(run_response.content), BaseModel):
            return f"Agent {member_name}: {run_response.content.model_dump_json(indent=2)}"
        return f"Agent {member_name}: {json.dumps(run_response.content, indent=2)}"
    except Exception as exc:
        return f"Agent {member_name}: Error - {str(exc)}"


def get_delegate_task_function(
    agent: Agent,
    run_response: RunOutput,
    run_context: RunContext,
    session: AgentSession,
    user_id: Optional[str] = None,
    stream: bool = False,
    stream_events: bool = False,
    async_mode: bool = False,
    input: Optional[str] = None,
    images: Optional[Sequence[Image]] = None,
    videos: Optional[Sequence[Video]] = None,
    audio: Optional[Sequence[Audio]] = None,
    files: Optional[Sequence[File]] = None,
    add_dependencies_to_context: Optional[bool] = None,
    add_session_state_to_context: Optional[bool] = None,
    debug_mode: Optional[bool] = None,
) -> Function:
    run_input = run_response.input
    _images = list(images or run_input.images if run_input and run_input.images else images or [])
    _videos = list(videos or run_input.videos if run_input and run_input.videos else videos or [])
    _audio = list(audio or run_input.audios if run_input and run_input.audios else audio or [])
    _files = list(files or run_input.files if run_input and run_input.files else files or [])

    def _setup_member(member: Member, task: str) -> tuple[Union[str, List[Message], None], Dict[str, Any]]:
        initialize_member(agent, member, debug_mode=debug_mode)
        if not agent.send_media_to_model:
            member.send_media_to_model = False
        member_task = input if agent.determine_input_for_members is False else task
        history: Optional[List[Message]] = None
        if hasattr(member, "add_history_to_context") and member.add_history_to_context:
            history = _get_history_for_member(session, member)
            if history and isinstance(member_task, str):
                history.append(Message(role="user", content=member_task))
        member_session_state = copy(run_context.session_state) if run_context.session_state is not None else {}
        return (history if history else member_task), member_session_state

    def _process_member_run(member_run_response: Optional[Union[RunOutput, TeamRunOutput]], member: Member) -> None:
        if member_run_response is None:
            return
        member_run_response.parent_run_id = run_response.run_id
        if run_response.tools is not None:
            for tool in run_response.tools:
                if tool.tool_name and tool.tool_name.lower() in {"delegate_task_to_member", "delegate_task_to_members"}:
                    tool.child_run_id = member_run_response.run_id
        if run_response.requirements is None and getattr(member_run_response, "requirements", None):
            run_response.requirements = []
        if getattr(member_run_response, "requirements", None):
            for requirement in member_run_response.requirements or []:
                requirement.member_agent_id = requirement.member_agent_id or get_member_id(member)
                requirement.member_agent_name = requirement.member_agent_name or member.name
                requirement.member_run_id = requirement.member_run_id or member_run_response.run_id
                run_response.requirements.append(requirement)
        session.upsert_run(member_run_response)  # type: ignore[arg-type]
        if run_context.session_state is None:
            run_context.session_state = {}
        merge_dictionaries(run_context.session_state, member_run_response.session_state or {})  # type: ignore[arg-type]

    def delegate_task_to_member(member_id: str, task: str) -> Iterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
        """Delegate a task to a selected member agent.

        Args:
            member_id: The ID of the member to delegate the task to.
            task: A clear task description and expected output for the member.
        """
        result = find_member_by_id(agent, member_id, run_context=run_context)
        if result is None:
            yield f"Member with ID {member_id} not found. Choose from:\n\n{get_members_system_message_content(agent, run_context=run_context)}"
            return

        _, member = result
        member_input, member_session_state = _setup_member(member, task)
        use_agent_logger()
        if stream:
            member_stream = member.run(  # type: ignore[union-attr]
                input=member_input,
                user_id=user_id,
                session_id=session.session_id,
                session_state=member_session_state,
                images=_images,
                videos=_videos,
                audio=_audio,
                files=_files,
                stream=True,
                stream_events=stream_events or agent.stream_member_events,
                debug_mode=debug_mode,
                dependencies=run_context.dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=run_context.metadata,
                knowledge_filters=run_context.knowledge_filters
                if not member.knowledge_filters and member.knowledge
                else None,
                yield_run_output=True,
            )
            member_run_response = None
            for item in member_stream:
                if isinstance(item, (RunOutput, TeamRunOutput)):
                    member_run_response = item
                    continue
                check_if_run_cancelled(item)
                item.parent_run_id = item.parent_run_id or run_response.run_id
                yield item  # type: ignore[misc]
        else:
            member_run_response = member.run(  # type: ignore[union-attr]
                input=member_input,
                user_id=user_id,
                session_id=session.session_id,
                session_state=member_session_state,
                images=_images,
                videos=_videos,
                audio=_audio,
                files=_files,
                stream=False,
                debug_mode=debug_mode,
                dependencies=run_context.dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=run_context.metadata,
                knowledge_filters=run_context.knowledge_filters
                if not member.knowledge_filters and member.knowledge
                else None,
            )
            check_if_run_cancelled(member_run_response)  # type: ignore[arg-type]

        _process_member_run(member_run_response, member)
        if not stream:
            yield _stringify_member_response(member_run_response, member.name or member.id or member_id)
        elif member_run_response is not None and member_run_response.is_paused:
            yield f"Agent {member.name}: Requires human input before continuing."

    async def adelegate_task_to_member(
        member_id: str, task: str
    ) -> AsyncIterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
        """Delegate a task to a selected member agent.

        Args:
            member_id: The ID of the member to delegate the task to.
            task: A clear task description and expected output for the member.
        """
        result = find_member_by_id(agent, member_id, run_context=run_context)
        if result is None:
            yield f"Member with ID {member_id} not found. Choose from:\n\n{get_members_system_message_content(agent, run_context=run_context, async_mode=True)}"
            return

        _, member = result
        member_input, member_session_state = _setup_member(member, task)
        use_agent_logger()
        if stream:
            member_stream = member.arun(  # type: ignore[union-attr]
                input=member_input,
                user_id=user_id,
                session_id=session.session_id,
                session_state=member_session_state,
                images=_images,
                videos=_videos,
                audio=_audio,
                files=_files,
                stream=True,
                stream_events=stream_events or agent.stream_member_events,
                debug_mode=debug_mode,
                dependencies=run_context.dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=run_context.metadata,
                knowledge_filters=run_context.knowledge_filters
                if not member.knowledge_filters and member.knowledge
                else None,
                yield_run_output=True,
            )
            member_run_response = None
            async for item in member_stream:
                if isinstance(item, (RunOutput, TeamRunOutput)):
                    member_run_response = item
                    continue
                check_if_run_cancelled(item)
                item.parent_run_id = item.parent_run_id or run_response.run_id
                yield item  # type: ignore[misc]
        else:
            member_run_response = await member.arun(  # type: ignore[union-attr]
                input=member_input,
                user_id=user_id,
                session_id=session.session_id,
                session_state=member_session_state,
                images=_images,
                videos=_videos,
                audio=_audio,
                files=_files,
                stream=False,
                debug_mode=debug_mode,
                dependencies=run_context.dependencies,
                add_dependencies_to_context=add_dependencies_to_context,
                add_session_state_to_context=add_session_state_to_context,
                metadata=run_context.metadata,
                knowledge_filters=run_context.knowledge_filters
                if not member.knowledge_filters and member.knowledge
                else None,
            )
            check_if_run_cancelled(member_run_response)

        _process_member_run(member_run_response, member)
        if not stream:
            yield _stringify_member_response(member_run_response, member.name or member.id or member_id)
        elif member_run_response is not None and member_run_response.is_paused:
            yield f"Agent {member.name}: Requires human input before continuing."

    async def adelegate_task_to_members(task: str) -> AsyncIterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
        """Delegate a task to all member agents and return their responses."""
        from agno.utils.callables import get_resolved_members

        resolved_members = get_resolved_members(agent, run_context) or []
        if stream:
            done_marker = object()
            queue: asyncio.Queue[Union[RunOutputEvent, TeamRunOutputEvent, str, object]] = asyncio.Queue()

            async def stream_member(member: Member) -> None:
                member_id = get_member_id(member) or member.name or ""
                async for item in adelegate_task_to_member(member_id, task):
                    await queue.put(item)
                await queue.put(done_marker)

            tasks = [asyncio.create_task(stream_member(member)) for member in resolved_members]
            completed = 0
            try:
                while completed < len(tasks):
                    item = await queue.get()
                    if item is done_marker:
                        completed += 1
                    else:
                        yield item  # type: ignore[misc]
            finally:
                for task_obj in tasks:
                    if not task_obj.done():
                        task_obj.cancel()
                for task_obj in tasks:
                    with contextlib.suppress(Exception, asyncio.CancelledError):
                        await task_obj
        else:
            tasks = []
            for member in resolved_members:
                member_id = get_member_id(member) or member.name or ""

                async def run_member(current_member_id: str = member_id) -> str:
                    responses = []
                    async for item in adelegate_task_to_member(current_member_id, task):
                        if isinstance(item, str):
                            responses.append(item)
                    return "\n".join(responses)

                tasks.append(run_member())
            for result in await asyncio.gather(*tasks):
                if result:
                    yield result

    def delegate_task_to_members(task: str) -> Iterator[Union[RunOutputEvent, TeamRunOutputEvent, str]]:
        """Delegate a task to all member agents and return their responses."""
        from agno.utils.callables import get_resolved_members

        resolved_members = get_resolved_members(agent, run_context) or []
        for member in resolved_members:
            member_id = get_member_id(member) or member.name or ""
            yield from delegate_task_to_member(member_id, task)

    if agent.delegate_to_all_members:
        return Function.from_callable(
            adelegate_task_to_members if async_mode else delegate_task_to_members,
            name="delegate_task_to_members",
        )
    return Function.from_callable(adelegate_task_to_member if async_mode else delegate_task_to_member)
