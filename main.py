import asyncio
import os

from dotenv import load_dotenv
from openai import AsyncOpenAI

from edlm_search.candidate import Candidate
from edlm_search.llm_pipeline import LLMPipeline
from edlm_search.problem import Problem
from edlm_search.runner import UnsafeRunner
from examples.et.evaluate import ETEvaluator

load_dotenv()


async def main():
    problem_dir = './examples/et'
    evaluator = ETEvaluator(problem_dir)
    runner = UnsafeRunner()

    llm_pipeline = LLMPipeline(
        async_openai=AsyncOpenAI(
            base_url='https://generativelanguage.googleapis.com/v1beta/openai/',
            api_key=os.environ['OPENAI_API_KEY'],
        ),
        model_name='gemini-2.5-flash',
    )
    problem = Problem.from_directory(problem_dir)
    candidate = await Candidate.new_from_problem(problem, llm_pipeline=llm_pipeline)
    print('candidate generated')

    metrics = await evaluator.evaluate(runner, candidate)
    print(f'Final metrics: {metrics}')


if __name__ == '__main__':
    asyncio.run(main())
