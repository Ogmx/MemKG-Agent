from typing import Optional, List
from pydantic import BaseModel, Field
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

# Two-shot w/context Prompts
few_shot_ctx_QA_instruction = """You are an expert to answer questions based on Knowledge Graph.
You need to give the correct answer based on the given triples and when the triples do not contain the correct answer, try to answer based on your own knowledge.
Here are some examples: """

few_shot_ctx_QA_examples = """
Question: where did flemish people come from
Triples: ['Flemish people', 'people.ethnicity.geographic_distribution', 'Australia']
Answer: France; Belgium; South Africa; Canada; Australia; United States of America; Brazil

Question: who did jackie robinson first play for
Triples: ['Jackie Robinson', 'sports.pro_athlete.teams', 'm.0ncxm4r'], ['m.0ncxm4r', 'sports.sports_team_roster.team', 'Montreal Royals'], ['m.0ncxm4r', 'sports.sports_team_roster.from', '?sk0']
Answer: UCLA Bruins football

"""

few_shot_ctx_QA_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", few_shot_ctx_QA_instruction + few_shot_ctx_QA_examples),
        ("user", "Question: {question}\nTriples: {triples}\n"),
    ]
)

# Two-shot w/o context Prompts
few_shot_noctx_QA_instruction = """You are an expert to answer questions based on Knowledge Graph.
You need to give the correct answer based on your own knowledge.
Here are some examples: """

few_shot_noctx_QA_examples = """
Question: where did flemish people come from
Answer: France; Belgium; South Africa; Canada; Australia; United States of America; Brazil

Question: who did jackie robinson first play for
Answer: UCLA Bruins football

"""

few_shot_noctx_QA_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", few_shot_noctx_QA_instruction + few_shot_noctx_QA_examples),
        ("user", "Question: {question}\n"),
    ]
)

# Prompt for entity extract
# copy from GoG：https://github.com/YaooXu/GoG/blob/main/prompts2/primitive_tasks/extract_entity

few_shot_entity_extract_instruction = """Please identify the entities in the given question.
You should answer these entities in list format directly.

Q: Which education institution has a sports team named George Washington Colonials men's basketball?
A: ["George Washington Colonials men's basketball"]

Q: What year did Baltimore Ravens win the superbowl?
A: ["Baltimore Ravens", "superbowl"]

Q: What countries share borders with Spain?
A: ["Spain"]
"""

few_shot_entity_extract_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", few_shot_entity_extract_instruction),
        ("user", "Question: {question}\n"),
    ]
)

# 'EXPAND_KG' - Search for edges of entities in the knowledge graph using an external API. This is a useful function for getting to the correct answer.
# \te.g.
# \t\tEXPAND_KG: I should search for the country of origin of Jonathon Taylor.
# \t\tEXPAND_KG: I need to search George Washington Colonials men's basketball, find the education institution that has it.
# 'ANSWER' - The knowledge graph (i.e. Knowledge Graph Entities and Knowledge Graph Edges) has enough information to generate the correct answer.
# \te.g. 
# \t\tANSWER
# \t\tANSWER: Harry Potter
# \t\tANSWER: [Harry Potter, Ron Weasley, Hermione Granger]
# 'THINK' - Based on the current state, generate thoughts and plans for solving the problem, including recalling known facts from memory.
# \te.g.
# \t\tTHINK: I should search for the movies directed by...
# \t\tTHINK: I know that Biden is American, therefore...
# \t\tTHINK: I see that John Cena is in both The Independent and Vacation Friends, therefore...
# The searched knowledge graph (i.e. Knowledge Graph Entities and Knowledge Graph Edges) has enough information to generate the correct answer.

ToT_ACTION_DESCRIPTIONS_TEXT_NO_FILTER = """You are a superintelligent AI equipped with the ability to search a knowledge graph for definitive, \
up-to-date answers. Your task is to explore the knowledge graph to answer the above questions.
You can choose from the following actions to explore or process KG to find the correct answer. Think in detail before acting.

Available actions:
\t'EXPAND_KG' - Explore the external Knowledge Graph to get more information. This is a helpful action for getting to the correct answer.
\t'ANSWER' - Use only if the answer of Original Query appears explicitly in the current Knowledge Graph Entities or Edges.

First think carefully about how the current knowledge graph relates to the original question (or sub-question).
Then choose an action from Available actions.

E.g. 
THINK:...
"""

ToT_ACTION_DESCRIPTIONS_TEXT = """You are a superintelligent AI equipped with the ability to search a knowledge graph for definitive, \
up-to-date answers. Your task is to explore the knowledge graph to answer the above questions.
You can choose from the following actions to explore or process KG to find the correct answer. Think in detail before acting.

Available actions:
\t'EXPAND_KG' - Explore the external Knowledge Graph to get more information. This is a helpful action for getting to the correct answer.
\t'FILTER' - Extract the exact information from the current ent_sets or perform logical operations on them by using six functions: [FilterbyCondition, FilterbyStr, LogicOperation, FindRelation, Count, Verify]. This is a useful action for obtaining valid information from ent_sets.
\t'ANSWER' - Use only if the answer of Original Query appears explicitly in the current Knowledge Graph Entities or Edges.

First think carefully about how the current knowledge graph relates to the original question (or sub-question).
Then choose an action from Available actions.

E.g. 
THINK:...
"""


ToT_FEW_SHOT_ACTION_DESCRIPTIONS_TEXT = """You are a superintelligent AI equipped with the ability to search a knowledge graph for definitive, \
up-to-date answers. Your task is to interface with the knowledge graph in order to answer the above query. \
You will be able to expand the knowledge graph until you have found the answer. Think in detail before acting or answering.\

Available actions:
\t'EXPAND_KG' - Search for edges of entities in the knowledge graph using an external API. This is a useful function for getting to the correct answer.
\t\tExample 1:
\t\t\tOriginal Query: What state is the the education institution has a sports team named George Washington Colonials men's basketball in?
\t\t\tKnowledge Graph Entities:
\t\t\tm.03d0l76: George Washington Colonials men's basketball
\t\t\tEXPAND_KG: I need to search George Washington Colonials men's basketball, find the education institution that has it.

\t\tExample 2:
\t\t\tOriginal Query: What state is the the education institution has a sports team named George Washington Colonials men's basketball in?
\t\t\tKnowledge Graph Edges:
\t\t\t(George Washington Colonials men's basketball, school_sports_team.school, George Washington University)
\t\t\t(George Washington Colonials men's basketball, sports_team.sport, basketball)
\t\t\t(George Washington Colonials, athletics_brand.teams, George Washington Colonials men's basketball)
\t\t\tEXPAND_KG: George Washington University has the team named George Washington Colonials men's basketball, so I need to find out which state is George Washington University in.

\t'ANSWER' - Generate the final answer once the problem is solved. Just state the best answer, do not output a full sentence. Attention please, entities begin with "m." (e.g., m.01041_p3) represent CVT (compound value type) node, and they shouldn't be selected as the final answers. To find out those entities involved in these event, you could select them as the entities to be searched.
\t\tExample 1:
\t\t\tOriginal Query: What state is the the education institution has a sports team named George Washington Colonials men's basketball in?

\t\t\tKnowledge Graph Edges:
\t\t\t(George Washington Colonials men's basketball, school_sports_team.school, George Washington University)
\t\t\t(George Washington Colonials men's basketball, sports_team.sport, basketball)
\t\t\t(George Washington Colonials, athletics_brand.teams, George Washington Colonials men's basketball)
\t\t\t(2003 DC Asian Pacific American Film Festival, event.locations, Washington, D.C.)
\t\t\t(George Washington University, location.located_in, Washington, D.C.)

\t\t\tTHOUGHT: So the answer is Washington, D.C.
\t\t\tANSWER: Washington, D.C.

\t\tExample 2:
\t\t\tOriginal Query: Which school with the fight song The Orange and Blue did Emmitt Smith play for?

\t\t\tKnowledge Graph Edges:
\t\t\t(The Orange and Blue, fight_song.sports_team, Florida Gators football)
\t\t\t(Florida Gators football, sports_team.fight_song, The Orange and Blue)
\t\t\t(The Orange and Blue, topic.notable_for, g.125b2pqsh)
\t\t\t(Emmitt Smith, person.profession, American football player)

\t\t\tTHOUGHT: Only one team's fight song is The Orange and Blue, so the final answer is Florida Gators football.
\t\t\tANSWER: Florida Gators football
"""
ToT_BASE_ACTION_SELECTION_PROMPT_NO_FILTER = ToT_ACTION_DESCRIPTIONS_TEXT_NO_FILTER + """SELECT ACTION {options}:..."""
ToT_BASE_ACTION_SELECTION_PROMPT = ToT_ACTION_DESCRIPTIONS_TEXT + """SELECT ACTION {options}:..."""

ToT_FULL_PROMPT = """{query}

{planning}
{kg_state}

Previous actions:
{trajectory}

{current_prompt}"""





ToT_DECOMPOSE_PROMPT = """
You are an expert in solving knowledge graph-based Q&A.
Your current task is to break down each constraint in the original query into independent simple sub-questions.

You need to make sure that each sub-problem is simple enough that can be answered by one-hop query (one triple) in the knowledge graph.
You must based solely on the original question and not introduce any guesswork or reasoning.
You need to make sure that at least one topic entity or word from original query is used in each sub-question.
You need to make sure that the answer to the original query can be obtained by solving the sub-problems step by step.
If the original query does not need to be decomposed, please output the original question directly instead of guessing.

First think carefully about what the original question is really asking and how to decompose that question into simple sub-questions.
Then, give the list of sub-questions.

E.g. 
THINK:...
Sub-Questions: 
1.
...
"""

ToT_EXPAND_KG_ENTITY_SELECTION_PROMPT = """
Objective: {think}

Your current task is to select an entity_set or entities from the KG to expand to get more information about.

If you want to explore a specific entity, select exactly one entity from the following entity_ids: {entities}
If you want to explore more than one entity or a set of entities, select from the following entity sets: {ent_sets}

You MUST select exactly one ent_set or entity_id from those listed above.

E.g. 
SELECT ENTITIES:"""

ToT_EXPAND_KG_RELATION_SELECTION_PROMPT = """
Objective: {think}

Your current task is to select a relation to expand along for the selected entities.
The selected entities are: {selected_entities}

The options of relations to choose from are:
{relations}

First think carefully about how to choose an appropriate relation to get close to the correct answer of original question / sub-questions / Objective.

Then, select exactly one relation from the options listed above.

E.g. 
THINK:...
SELECT RELATION:...
"""
# In addition, you can add constraints to make the acquired entities more specific.
# A constraint consists of a logical operator O and a string V. 
# O should be one of [=, >, >=, <, <=, argmax, argmin], which means the comparison between the acquired entities and string value should satisfy the operator.
# V can be a concept name, type, number, date, year, etc.

# By adding constraints, you can create a query graph like this: (selected entities, selected relation, ?ent) (?ent, O, V)
#CONSTRAIN:[O,V]

# - FilterbyType
# Description - Filter entities belonging to (is instance of) the given type / concept.
# Inputs - type / concept name
# - FilterbyStr
# Description - Filter out entities whose relations or attributes contain a particular string, i.e. find the <entities> whose <attribute key OR attribute value OR relation> contain <string>.
# Inputs - string
# - FilterbyRelation
# Description - Filter entities that have a specific relation with the given entity, i.e. find the <entities> have <relation> of <entity>.
# Inputs - (relation, entity)



ToT_FILTER_ENTITY_PROMPT = """{obs}\n
Objective: {think}

Your current task is to filter out valid information from the current knowledge graph by using a list of functions.

List of Functions:

- FilterbyStr
Description - Filter out entities related to the given string, i.e., one-hop neighborhoods containing the string. Wide range filtering when exact conditions are uncertain.
Inputs - (string)

- FilterbyCondition
Description - Filtering entities that have specific edge and its corresponding value satisfy a specific condition, i.e. find the <entities> whose <key / relation> is <operator> <value>.
Input - (key, operator, value); the operator should be one of  [=, !=, <, <=, >, >=, argmax, argmin], The value should be a string or "ANY" when there is no value.

- FindRelation
Description - Find the relation / connection of two specific entities, i.e. get the <relation> of <entity_1> and <entity_2>.
Input - (entity_1, entity_2)

- LogicOperation
Description - Processing a list of ent_sets with a specific logical operation, i.e. conduct <operator> on entity set list [<entity_set_id>].
Input - (operator, [ent_set_xx, ent_set_xx, ...]), the operator should be one of [intersect, union]

- Count
Description - Similar to FilterbyCondition, but only returns the number of entities that satisfy the condition, i.e. get the number of <entities> whose <key> is <operator> <value>.
Input - (key, operator, value), the operator should be one of  [=, !=, <, <=, >, >=]

- Verify
Description - Similar to FilterbyCondition, but only returns a boolean value indicating whether any entity meets the given conditions, i.e. For <entities>, is his/her/its <key> <operator> <value>.
Input - (key, operator, value), the operator should be one of  [=, !=, <, <=, >, >=]


First think carefully about what information has been filtered out and how choosing the right function helps to answer the original question/sub-question.
Then select ent_set(s) to process on from: {entities}
Finally, select a function and give an input that meets the formatting requirements of that function.

Do not use the same functions and inputs that have appeared in Previous actions!

e.g.
THINK: ...
Select Entity Set(s): ...
Select Function: ...
Function Input: (...)
"""


ToT_EXTRACT_PROMPT = """
Objective: {think}

Your current task is to find the answer to the original query from the above knowledge graph.

If the answer appears directly in Knowledge Graph Edges, select the corresponding edges.
If the answer appears in Knowledge Graph Entities, select the corresponding ent_sets or entities.
If the question cannot be answered based on the current KG information, select "NO ANS".
Note the direction of the triples, if you are not sure which is the correct direction, select more than one.

First think carefully about how the current knowledge graph relates to the original question and where to extract the valid information.
Then, double-check if each constraint in the original query is satisfied by the corresponding ent_set or edge in above KG.
Finally, if all constraints are satisfied, select the ent_sets or edges that contain the final answer.

e.g.
THINK: ...
CHECK:
  constraint 1: which ent_set or edge satisfies it ...
  constraint 2: which ent_set or edge satisfies it ...
  constraint 3: ...
  ...
SELECT: ent_sets or edges or NO ANS
"""

ToT_ANSWER_PROMPT = """Objective: {think}

This information extracted from the current KG may be helpful: {hint}

Give your best answer based on the knowledge graph. If the knowledge graph do not contain the correct answer, try to answer based on your own knowledge.
First think carefully about what the original question is asking and which triples or entities contain the answer. 
The correct answer will usually appear explicitly in a Knowledge Graph Edge or ent_set containing no more than 5 entities.
Pay attention to the direction of the triples to make sure they are consistent with the question.

If the question is a judgment or comparison, answer “Yes” or “No”. If the question asks for a specific number or count, answer with a specific number.
If the answer appears directly in the Knowledge Graph please give its IDs, which starts with m.xxx or Q...

e.g.
THINK: ...
ANSWER: ...
ANSWER_IDs: ... 
"""

SIMPE_ANSWER_PROPMT = """You are an assistant for question-answering tasks. Use the following triples from a knowledge graph to answer the question. Only answer the entity name without any additional information.
Triples: {triples}
Question: {question}
Answer:..."
"""

SIMPE_QA_PROPMT = """You are an assistant for question-answering tasks. Try your best to answer the following question. Only answer the entity name without any additional information.
Question: {question}
Answer:..."
"""


ToT_EVALUATE_STATE_PROMPT = """Your current task is to evaluate the current state, which contains the retrieved knowledge graph(KG) and pervious actions.
You must scrutinize the current KG and previous actions to help decide whether to continue along the current state or redo.
Give a pessimistic score from 0.0 to 1.0 on how likely along the current state will result in a final correct answer.

1.0 if the current KG contains the answer to the original question explicitly.
0.8 if the answer to the original question is contained in any ent_set or triple of the current KG but does not appear explicitly.
0.8 if the current KG contains a partial answer to the original question (or any sub-questions), and the pervious actions do not contain many invalid actions.
0.6 if the current state may be in the correct state chain and worth further exploration, and the pervious actions do not contain many invalid actions.
0.4 if the current state may be in the correct state chain but the the pervious actions contain many invalid actions / noise information.
0.2 if most of the pervious actions are unhelpful and unrelated to answer the original question or sub-questions.
0.0 if the current state is completely wrong, contains no information relevant to the question, or all previous actions are unhelpful in answering the question.

First think about the relationship between the current knowledge graph and the original question (or sub-questions).
Then give a score about how likely the current state is one of the correct chain of states.

E.g. 
THINK:...
SCORE [0.0-1.0]:..."""


ToT_EVALUATE_ANSWER_PROMPT = """
Provided answer: {answer}

Your task is to score the correctness of the provided answer based on the original query, and the knowledge graph.
Double-check that each constraint in the original query is satisfied by the edges or ent_sets in KG.
Give a pessimistic score from 0.0 to 1.0 on how likely the answer is to be correct. 

0.0 if definitely wrong or avoids answering the question
0.0 if unable to answer based on the knowledge graph
0.3 if not all constraints (sub-questions) in the original query are satisfied or only some of them are satisfied
0.5 if unsure
0.7 for probably correct but not confirmed in knowledge graph
1.0 for definitely correct and confirmed in knowledge graph

First think carefully about what the original question is asking and whether the current knowledge graph fulfills all of the requirements (each sub-question) in the original question.
Then give a score about how likely the provided answer is to be correct.

E.g. 
THINK:...
SCORE:...
"""

GoG_thought_instruction = """Solve a question answering task with interleaving Thought, Action, Observation steps. Thought can reason about the current situation, and Action can be three types: 
(1) EXPAND_KG[entity1 | entity2 | ...], which searches the exact entities on Freebase and returns their one-hop subgraphs. You should extract the all concrete entities appeared in your last thought without redundant words.
(2) ANSWER[answer1 | answer2 | ...], which returns the answer and finishes the task. The answers should be complete entity label appeared in the triples. If you don't know the answer, please output Finish[unknown]. 
(3) REFLECTION[thought], which generate relevant thoughts or triples to solving the problem. This could include recalling well known facts from your inherent knowledge directly or reasoning from the given triples.
Entities and answers should be separated by tab.
Attention please, entities begin with "m." (e.g., m.01041_p3) represent CVT (compound value type) node, and they shouldn't be selected as the final answers. To find out those entities involved in these event, you could select them as the entities to be searched.
You should generate each step without redundant words.

Here are some examples.
"""

GoG_thought_examples="""Question: What state is the the education institution has a sports team named George Washington Colonials men's basketball in?
Topic Entity: [George Washington Colonials men's basketball]
Thought 1: I need to search George Washington Colonials men's basketball, find the education institution that has it.
Action 1: EXPAND_KG[George Washington Colonials men's basketball]
Observation 1: George Washington Colonials men's basketball, school_sports_team.school, George Washington University
George Washington Colonials men's basketball, sports_team.sport, basketball
George Washington Colonials, athletics_brand.teams, George Washington Colonials men's basketball
Thought 2: George Washington University has the team named George Washington Colonials men's basketball, so I need to find out which state is George Washington University in.
Action 2: EXPAND_KG[George Washington University]
Observation 2: George Washington University, organization.headquarters, m.0cnnz2d
George Washington University, location.contains, Madison Hall
George Washington University, location.events, 2003 DC Asian Pacific American Film Festival
Thought 3: There is no information about which state is George Washington University in, so I need to generate more triples based on these given triples and my inherent knowledge.
Action 3: REFLECTION[which state is George Washington University in]
Observation 3: 2003 DC Asian Pacific American Film Festival, event.locations, Washington, D.C.
George Washington University, location.located_in, Washington, D.C.
Thought 4: So the answer is Washington, D.C.
Action 4: ANSWER[Washington, D.C]

Question: Which school with the fight song The Orange and Blue did Emmitt Smith play for?
Topic Entity: [Emmitt Smith | The Orange and Blue]
Thought 1: I need to find football teams related to both emmitt smith and Orange and Blue.
Action 1: EXPAND_KG[Emmitt Smith | The Orange and Blue]
Observation 1: The Orange and Blue, fight_song.sports_team, Florida Gators football
Florida Gators football, sports_team.fight_song, The Orange and Blue
The Orange and Blue, topic.notable_for, g.125b2pqsh
Emmitt Smith, person.profession, American football player
Emmitt Smith, person.profession, Athlete
Emmitt Smith, pro_athlete.teams, m.0hqd__y
Emmitt Smith, pro_athlete.teams, m.0hqf002
Emmitt Smith, pro_athlete.teams, m.0hqf007
Thought 2: Only one team's fight song is The Orange and Blue, so the final answer is Florida Gators football, 
Action 2: ANSWER[Florida Gators football]
"""

GoG_BASE_PROMPT = GoG_thought_instruction + GoG_thought_examples + "Question: {question}\nTopic Entity: {topic_entity_names_str}\n"


prompt_dict = {
    "few_shot_ctx_QA_prompt": few_shot_ctx_QA_prompt,
    "few_shot_noctx_QA_prompt": few_shot_noctx_QA_prompt,
    "few_shot_entity_extract_prompt": few_shot_entity_extract_prompt,
    "SIMPE_QA_PROPMT": SIMPE_QA_PROPMT,
    "SIMPE_ANSWER_PROPMT": SIMPE_ANSWER_PROPMT,
    "GoG_base_prompt": GoG_BASE_PROMPT,
    "ToT_base_prompt": ToT_BASE_ACTION_SELECTION_PROMPT,
    "ToT_full_prompt": ToT_FULL_PROMPT,
    "ToT_expand_ent_prompt": ToT_EXPAND_KG_ENTITY_SELECTION_PROMPT,
    "ToT_expand_rel_prompt": ToT_EXPAND_KG_RELATION_SELECTION_PROMPT,
    "ToT_answer_prompt": ToT_ANSWER_PROMPT,
    "ToT_EVALUATE_STATE_PROMPT": ToT_EVALUATE_STATE_PROMPT,
    "ToT_EVALUATE_ANSWER_PROMPT": ToT_EVALUATE_ANSWER_PROMPT,
    "ToT_FILTER_ENTITY_PROMPT": ToT_FILTER_ENTITY_PROMPT,
    "ToT_DECOMPOSE_PROMPT":ToT_DECOMPOSE_PROMPT,
    "ToT_EXTRACT_PROMPT": ToT_EXTRACT_PROMPT,
    "ToT_BASE_ACTION_SELECTION_PROMPT_NO_FILTER": ToT_BASE_ACTION_SELECTION_PROMPT_NO_FILTER,
}

#################### Output Format ####################
class base_QA_format(BaseModel):
    """Answering a question"""

    answers: str = Field(description="Answers to the user question contain only entity names, if there is more than one answer split with a semicolon")

class entity_extract_format(BaseModel):
    """Identify the entities in the given question"""

    entities: List[str] = Field(description="Entities included in the question")

# for KG-Agent
class agent_think_format(BaseModel):
    """Select an action from the Next action options"""

    action_name: str = Field(description="Specific action name selected from the next action options")
    action_description: str = Field(description="A sentence to explain the selected action")

class agent_expand_ent_format(BaseModel):
    """Select up to three entities from the given options to explore"""

    entities: List[str] = Field(description="Selected Entities")

class agent_expand_rel_format(BaseModel):
    """Select one relation from the given options to explore"""

    relation: str = Field(description="Selected relation")

class agent_answer_format(BaseModel):
    """Give your best answer based on the knowledge graph"""

    answers: List[str] = Field(description="A list of answers to the Original Query contain only entity names")

output_format_dict = {
    "base_QA_format" : base_QA_format,
    "entity_extract_format": entity_extract_format,

    "agent_think_format": agent_think_format,
    "agent_expand_ent_format": agent_expand_ent_format,
    "agent_expand_rel_format": agent_expand_rel_format,
    "agent_answer_format": agent_answer_format,
}

################### Utils ##########################
def load_result(df):
    results = [df.iloc[i]['output'].split("answers=")[-1].replace("'","") for i in range(len(df))]
    return results