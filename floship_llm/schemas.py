
from pydantic import BaseModel, Field
from typing import Optional, List

class ThinkingModel(BaseModel):
    # A base model for representing a thought process or reasoning for LLM.
    thinking: str = Field(
        ...,
        title="Thought Process",
        description="A string representing the thought process, chain of thought, or reasoning behind the action taken."
    )
    
class Suggestion(ThinkingModel):
    file_path: str = Field(description="The path to the file where the suggestion is made.")
    line: int = Field(description="The line number in the file where the suggestion is made.")
    suggestion: str = Field(description="The suggested code change in a markdown code block using the ```suggestion marker.")
    severity: int = Field(description="The severity of the suggestion, 0 - low, 10 - high.")
    type: str = Field(description="The type of suggestion, one of the following: 'bug', 'neatpick', 'text_change', 'refactor', 'performance', 'security'.")
    reason: str = Field(description="The reason for the suggestion, a markdown text explaining why the change is needed.")

class SuggestionsResponse(ThinkingModel):
    suggestions: List[Suggestion] = Field(description="A list of code change suggestions.")
    
    
class Labels(ThinkingModel):
    # A model for representing labels for a jira ticket.
    labels: List[str] = Field(
        ...,
        title="Labels",
        description="A list of unique labels to be added to the jira ticket. Not more than 5 labels."
    )
