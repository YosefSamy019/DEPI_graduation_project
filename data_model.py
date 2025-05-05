from typing import Literal, List
from pydantic import BaseModel, Field
import json

_targeted_lang = "English"

_StoryCategory = Literal[
    "politics", "sports", "art", "technology", "economy",
    "health", "entertainment", "science", "not_specified"
]

_EntityType = Literal[
    "person-male", "person-female", "location", "organization", "event", "time",
    "quantity", "money", "product", "law", "disease", "artifact", "not_specified"
]

class Entity(BaseModel):
    entity_value: str = Field(..., description="The actual name or value of the entity.")
    entity_type: _EntityType = Field(..., description="The type of recognized entity.")


class NewsDetails(BaseModel):
    story_title: str = Field(..., min_length=5, max_length=300,
                             description="A fully informative and SEO optimized title of the story.")

    story_keywords: List[str] = Field(..., min_items=1,
                                      description="Relevant keywords associated with the story.")

    story_summary: List[str] = Field(
                                    ..., min_items=1, max_items=5,
                                    description="Summarized key points about the story (1-5 points)."
                                )

    story_category: _StoryCategory = Field(..., description="Category of the news story.")

    story_entities: List[Entity] = Field(..., min_items=1, max_items=10,
                                        description="List of identified entities in the story.")
    


class TranslatedStory(BaseModel):
    translated_title: str = Field(..., min_length=5, max_length=300,
                                  description="Suggested translated title of the news story.")
    translated_content: str = Field(..., min_length=5,
                                    description="Translated content of the news story.")



def getDetailExtractionMessage(story:str):
    details_extraction_message = [
        {
            "role": "system",
            "content": "\n".join([
                "You are an NLP data paraser.",
                "You will be provided by an Arabic text associated with a Pydantic scheme.",
                "Generate the ouptut in the same story language.",
                "You have to extract JSON details from text according the Pydantic details.",
                "Extract details as mentioned in text.",
                "Do not generate any introduction or conclusion."
            ])
        },
        {
            "role": "user",
            "content": "\n".join([
                "## Story:",
                story.strip(),
                "",

                "## Pydantic Details:",
                json.dumps(
                    NewsDetails.model_json_schema(), ensure_ascii=False
                ),
                "",

                "## Story Details:",
                "```json"
            ])
        }
    ]
        
    return details_extraction_message

def getTranslationMessage(story:str):
    translation_message = [
        {
            "role": "system",
            "content": "\n".join([
                "You are a professional translator.",
                "You will be provided by an Arabic text.",
                "You have to translate the text into the `Targeted Language`.",
                "Follow the provided Scheme to generate a JSON",
                "Do not generate any introduction or conclusion."
            ])
        },
        {
            "role": "user",
            "content":  "\n".join([
                "## Story:",
                story.strip(),
                "",

                "## Pydantic Details:",
                json.dumps( TranslatedStory.model_json_schema(), ensure_ascii=False ),
                "",

                "## Targeted Language:",
                _targeted_lang,
                "",

                "## Translated Story:",
                "```json"

            ])
        }
    ]
    
    return translation_message