import asyncio
import os
from dotenv import load_dotenv
from edlm_search.candidate import Candidate

load_dotenv()

from openai import AsyncOpenAI
from edlm_search.llm_pipeline import LLMPipeline
from edlm_search.problem import Problem


async def main():
    print(os.environ['OPENAI_API_KEY'])
    problem = Problem.from_directory('./examples/eth')
    llm_pipeline = LLMPipeline(
        async_openai=AsyncOpenAI(
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
            api_key=os.environ['OPENAI_API_KEY'],
        ),
        model_name='gemini-2.5-pro',
    )
    candidate = await Candidate.from_problem(problem, llm_pipeline=llm_pipeline)


asyncio.run(main())
