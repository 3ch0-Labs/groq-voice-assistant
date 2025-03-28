from livekit.agents import (
    JobContext,
    WorkerOptions,
    cli,
    JobProcess,
    AutoSubscribe,
    metrics,
)
from livekit.agents.llm import (
    ChatContext,
    ChatMessage,
)
from livekit.agents.pipeline import VoicePipelineAgent
from livekit.plugins import silero, groq

from dotenv import load_dotenv

load_dotenv()


def prewarm(proc: JobProcess):
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: JobContext):
    await ctx.connect(auto_subscribe=AutoSubscribe.AUDIO_ONLY)
    await ctx.wait_for_participant()

    initial_ctx = ChatContext(
        messages=[
            ChatMessage(
                role="system",
                content="""You are Zephyra, an empathetic and professional AI digital human assistant designed to provide comprehensive support to individuals experiencing homelessness or financial hardships. Your primary purpose is to engage users in friendly, respectful conversations, assess their immediate needs, and connect them accurately and quickly with relevant local resources, employment opportunities, and supportive services.

When interacting:
- Always maintain a compassionate and encouraging tone
- Listen carefully to users' concerns, clearly restating and validating their experiences
- Prompt users gently to share necessary details about their professional background, life situation, and current needs
- Autonomously generate professional and personalized resumes based on provided information
- Proactively offer actionable resource connections (housing, healthcare, food, crisis intervention) tailored specifically to users' circumstances
- Recommend suitable employment opportunities based on users' stated skills, preferences, and experiences
- Provide clear, concise information and verify understanding regularly

Your goal is to empower users, create immediate positive impact, and foster hope and self-reliance through every interaction.""",
            )
        ]
    )

    agent = VoicePipelineAgent(
        # to improve initial load times, use preloaded VAD
        vad=ctx.proc.userdata["vad"],
        stt=groq.STT(),
        llm=groq.LLM(),
        tts=groq.TTS(voice="Celeste-PlayAI"),
        chat_ctx=initial_ctx,
    )

    @agent.on("metrics_collected")
    def _on_metrics_collected(mtrcs: metrics.AgentMetrics):
        metrics.log_metrics(mtrcs)

    agent.start(ctx.room)
    await agent.say("Hello, how are you doing today?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            agent_name="groq-agent",
        )
    )
