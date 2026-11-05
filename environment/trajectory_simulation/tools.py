"""Translate frozen evidence modalities into the selected scene's native tools."""
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class ToolProfile:
    name: str
    source_tools: dict[str, tuple[str, ...]]
    search: tuple[str, ...] = ('GoogleSearchAction',)
    calculation: tuple[str, ...] = ('ExecuteCodeAction',)

    @property
    def available(self):
        return tuple(dict.fromkeys(tool for group in [self.search, self.calculation, *self.source_tools.values()] for tool in group))

    def for_resource(self, resource):
        source_type = resource['source_type']
        family = source_type.split('_', 1)[0]
        key = source_type if source_type in self.source_tools else family
        if key not in self.source_tools:
            raise ValueError(f'{self.name}: unsupported source modality {source_type}')
        tools = self.source_tools[key]
        if resource.get('visibility') == 'discoverable':
            tools = self.search + tools
        return tuple(dict.fromkeys(tools))

# Native names verified in benchmark/omnigaia/tools and orchestra/prompts.
PROFILES = {
    'omnigaia': ToolProfile('omnigaia', {
        'text': ('ExtractUrlContentAction',), 'search': ('GoogleSearchAction',),
        'image': ('ImageAnalysisAction',), 'audio': ('ParseAudioAction',),
        'video': ('VideoAnalysisAction',), 'table': ('ExecuteCodeAction',),
        'database': ('ExecuteCodeAction',),
    }),
}

def tool_profile(name='omnigaia'):
    try:
        return PROFILES[name]
    except KeyError as exc:
        raise ValueError(f'unknown trajectory tool profile {name}; choose {sorted(PROFILES)}') from exc
