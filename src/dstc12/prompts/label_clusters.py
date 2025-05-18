# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.  
# SPDX-License-Identifier: CC-BY-NC-4.0

from langchain_core.prompts import PromptTemplate


PROMPT = PromptTemplate.from_template(
'''<task>
You are an expert call center assistant. You will be given a set of utterances in <utterances> </utterances> tags, each one on a new line.
The utterances are part of callcenter conversations between the customer and the support agent.
Your task is to generate a short label describing the theme of all the given utterances. The theme label should be under 5 words and describe the desired customer's action in the call.


<guidance>
Output your response in the following way.
<theme_label_explanation>Your short step-by-step explanation behind the theme</theme_label_explanation>
<theme_label>your theme label</theme_label>
</guidance>
</task>

H:
<utterances>
{utterances}
</utterances>
'''
)