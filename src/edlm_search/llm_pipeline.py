import pathlib

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam
from bs4 import BeautifulSoup

here = pathlib.Path(__file__).parent.resolve()
jinja_env = Environment(
    loader=FileSystemLoader(str(here / 'prompts')), undefined=StrictUndefined
)


class LLMPipeline:
    def __init__(self, async_openai: AsyncOpenAI, model_name: str):
        self._async_openai = async_openai
        self._model_name: str = model_name

    async def generate_files_from_template(
        self, template_name: str, **kwargs
    ) -> dict[str, str]:
        template_name = template_name.removesuffix('.jinja').removesuffix('.md')
        prompt_template = jinja_env.get_template(f'{template_name}.md.jinja')

        prompt: ChatCompletionUserMessageParam = {
            'role': 'user',
            'content': prompt_template.render(**kwargs),
        }

        response = await self._async_openai.chat.completions.create(
            messages=[prompt], model=self._model_name
        )
        output = response.choices[0].message.content
        assert output

        print(output)
        soup = BeautifulSoup(output, 'xml')

        files_tag = soup.find('files')

        file_dict = {}
        if files_tag:
            for file_element in files_tag.find_all('file'):
                path = file_element.get('path')
                content = file_element.get_text(strip=True)
                if path:
                    file_dict[path] = content

        return file_dict
