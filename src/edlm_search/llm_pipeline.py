import pathlib
import re
from inspect import cleandoc

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionUserMessageParam

here = pathlib.Path(__file__).parent.resolve()
jinja_env = Environment(
    loader=FileSystemLoader(str(here / 'prompts')), undefined=StrictUndefined
)


class ModelOutputParseError(Exception):
    pass


class LLMPipeline:
    def __init__(self, async_openai: AsyncOpenAI, model_name: str):
        self._async_openai = async_openai
        self._model_name: str = model_name

    def _parse_xml_files(self, xml_string: str) -> dict[str, str]:
        """
        Parses an XML-like string containing file data and returns a dictionary.

        This parser is "forgiving" and uses regex to extract content,
        allowing special characters like '<' or '&' inside the file tags.

        Args
        -----
            xml_string: A string containing the data, expected to be wrapped in
                        <files>...</files> and contain <file path="...">...</file> tags.

        Returns
        -------
            A dictionary where keys are file paths and values are file contents.
        """
        # try prase cdata?
        # try parse markdown code blocks?
        files_dict = {}

        # This regex finds all <file> tags and captures two groups:
        # 1. ([^"]+): The content of the path="..." attribute.
        # 2. (.+?): The content inside the tag (non-greedy).
        # re.DOTALL (or re.S) is crucial so that '.' matches newline characters.
        pattern = re.compile(r'<file path="([^"]+)">(.+?)</file>', re.DOTALL)

        matches = list(pattern.finditer(xml_string))

        if not matches:
            # Add checks to provide better error messages
            if '<files>' not in xml_string or '</files>' not in xml_string:
                raise ModelOutputParseError('failed to find <files>...</files> section')
            raise ModelOutputParseError('no <file ...> entries found within <files> section')

        for match in matches:
            path = match.group(1)
            content = match.group(2)

            if not path:
                # This is unlikely with the regex, but good practice
                raise ModelOutputParseError('found a file entry with no path')

            # Use cleandoc to remove leading whitespace from the code block
            cleaned_content = cleandoc(content)

            if not cleaned_content.strip():
                raise ModelOutputParseError(f"content of file entry for '{path}' is empty")

            files_dict[path] = cleaned_content

        return files_dict

    async def generate_files_from_template(
        self, template_name: str, **kwargs
    ) -> tuple[str, dict[str, str]]:
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
        print(output)
        assert output

        idea_match = re.search(r'<idea>(.*?)</idea>', output, re.DOTALL)
        if not idea_match:
            raise ModelOutputParseError('failed to find <idea>...</idea> section')
        idea = idea_match.group(1).strip()

        files_match = re.search(r'<files>(.*?)</files>', output, re.DOTALL)
        if not files_match:
            raise ModelOutputParseError('failed to find <files>...</files> section')
        files_content = files_match.group(1)

        files_dict = self._parse_xml_files(files_content)

        return idea, files_dict
