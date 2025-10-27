def _prefix_language_hint(language_hint: str) -> str:
    return f"{language_hint}\n\n" if language_hint else ""


def _language_note(language: str, zh_message: str, en_message: str) -> str:
    return zh_message if language == "zh" else en_message


def get_report_webthinker_instruction(question, plan, language: str = "en"):
    language_guidance = _language_note(
        language,
        "请确保所有对外可见的内容（搜索意图、搜索查询、章节写作请求、编辑指令、文章正文以及大纲）主要使用简体中文撰写；必要的英文术语或缩写可直接保留。内部推理 <|begin_think|> … <|end_think|> 可以使用任意语言。",
        "Ensure every visible output (search intents, search queries, section requests, edit instructions, article body, and outline) is written in English; internal <|begin_think|> … <|end_think|> may use any language.",
    )

    example_block = _language_note(
        language,
        """<|begin_think|>目标：先获取 X 的权威定义；缺少：Y 的指标；下一步：搜索 X 的定义。<|end_think|>
<|begin_search_query|>
intent: 寻找 X 的权威定义
query: X 的机器学习定义
<|end_search_query|>
<|begin_search_result|>来自相关网页的摘要信息<|end_search_result|>
<|begin_think|>根据这些结果，我已经了解 X，但仍需调查 Y……<|end_think|>
<|begin_search_query|>
intent: 获取与 Y 相关的指标
query: Y 的评估指标
<|end_search_query|>
<|begin_search_result|>来自相关网页的摘要信息<|end_search_result|>
<|begin_think|>现在我已收集足够信息，可以撰写第一节……<|end_think|>
<|begin_write_section|>引言
本节应介绍……<|end_write_section|>
<|begin_think|>完成引言后，需要继续获取资料撰写下一节……完成上述章节后，要检查当前文章确保内容完整准确。<|end_think|>
<|begin_check_article|><|end_check_article|>
<|begin_think|>我意识到需要修改……<|end_think|>
<|begin_edit_article|>你的编辑指令<|end_edit_article|>""",
        """<|begin_think|>Goal: first obtain an authoritative definition of X; missing: metrics for Y; next step: search for X definition.<|end_think|>
<|begin_search_query|>
intent: find an authoritative definition of X
query: definition of X in machine learning
<|end_search_query|>
<|begin_search_result|>Summary of information from searched web pages<|end_search_result|>
<|begin_think|>Based on these results, I understand X, but still need to investigate Y...<|end_think|>
<|begin_search_query|>
intent: gather metrics related to Y
query: evaluation metrics for Y
<|end_search_query|>
<|begin_search_result|>Summary of information from searched web pages<|end_search_result|>
<|begin_think|>Now I have enough information to write the first section...<|end_think|>
<|begin_write_section|>Introduction
This section should introduce ... <|end_write_section|>
<|begin_think|>I have written the introduction. Now I need to explore more information to write the next section ...
After writing the above sections, I need to check the current article to ensure the content is complete and accurate.<|end_think|>
<|begin_check_article|><|end_check_article|>
<|begin_think|>Wait, I realize that I need to edit ...<|end_think|>
<|begin_edit_article|>your edit instruction<|end_edit_article|>""",
    )

    return f"""You are a research assistant with the ability to perform web searches to write a scientific research article. You have special tools:

- To perform a search: you must always enclose both the intent and the query strictly inside a matching pair of tags:
<|begin_search_query|>
intent: briefly describe why you are performing this search
query: your query here
<|end_search_query|>
The <|begin_search_query|> and <|end_search_query|> tags must always appear as a matching pair without omission.
Then, the system will search and analyze relevant web pages, then provide you with helpful information in the format <|begin_search_result|>search results<|end_search_result|>.

- To write a section of the research article: you must always enclose both the section name and the contents to write inside a matching pair of tags:
<|begin_write_section|>section name\\ncontents to write<|end_write_section|>.
The <|begin_write_section|> and <|end_write_section|> tags must always appear as a matching pair without omission.
Then, the system will completely write the section based on your request and current gathered information.

- To check the current article: write <|begin_check_article|><|end_check_article|>.
The <|begin_check_article|> and <|end_check_article|> tags must always appear as a matching pair without omission.
Then, the system will return the outline of all current written contents in the format <|begin_article_outline|>outline contents<|end_article_outline|>.

- To edit the article: write <|begin_edit_article|>your detailed edit goal and instruction<|end_edit_article|>.
The <|begin_edit_article|> and <|end_edit_article|> tags must always appear as a matching pair without omission.
Then, the system will edit the article based on your goal and instruction and current gathered information.

---

{language_guidance}

---

### Hidden Reasoning Protocol
- Always enclose your private planning, analysis of search results, and decision-making process inside <|begin_think|> … <|end_think|>. The <|begin_think|> and <|end_think|> tags must always appear as a matching pair without omission.
- Never leak or copy any <|begin_think|> content into visible outputs (article text, search queries, written sections, or edit instructions).
- <|begin_think|> is strictly for internal reasoning. If it appears in visible output, that is an error.

---

Your task is to research and write a scientific article about:
{question}

Here is a research plan to guide your investigation:
{plan}

Please follow the research plan step by step:
1. Use web searches to gather detailed information for each point
2. In <|begin_think|>, analyze each search result and determine what additional information is needed  
3. When you have sufficient information for a section, request to write that section
4. Continue this process until the full article is complete
5. Check the current article and edit sections as needed to improve clarity and completeness

Example:
{example_block}

Assistant continues gathering information and writing sections until getting comprehensive information and finishing the entire article.

Remember:
- Always use <|begin_search_query|>intent + query<|end_search_query|> with both begin and end tags strictly paired to perform web searches.
- Always use <|begin_write_section|>section name\\ncontents to write<|end_write_section|> to call the system to write a section in the article
- Always use <|begin_check_article|><|end_check_article|> to check the current written article
- Always use <|begin_edit_article|>edit instruction<|end_edit_article|> to call the system to edit and improve the article
- You should strictly follow the above format to call the functions.
- Do not propose methods or design experiments, your task is to comprehensively research with web searches.
- Do not omit any key points in the article.
- When you think the article is complete, directly output "I have finished my work." and stop.

Now begin your research and write the article about:
{question}
"""


def get_search_plan_instruction(
    query, max_tool_calls=5, max_plan_steps=8, language: str = "en"
):
    language_requirement = _language_note(
        language,
        "- 步骤请主要使用简体中文撰写，可保留必要的英文术语或缩写。\n",
        "- Write every step in English.\n",
    )
    return f"""You are an expert research planner. 
Your task is to create a detailed, low-cost web search plan for answering the following user question:

{query}

Goals:
- Deliver a complete, reliable answer with as few web calls as possible.
- First, attempt to answer from internal knowledge, memory, or cache with a confidence estimate.
- Only use web search if confidence is low, ambiguity remains, or authoritative/updated data is required.
- Always stop early if enough authoritative information has been found.

Hard constraints:
- At most {max_tool_calls} search queries total (strict budget).
- Merge multiple intents into one query whenever possible, using site: filters and OR groups.
- Avoid duplicate domains or redundant opens.
- Stop early once authoritative sources provide sufficient coverage.

The plan must include:
1. Ambiguity resolution: clarify what exactly needs to be measured, compared, or defined.
2. Draft from internal knowledge: attempt a tentative answer from memory/cache and state confidence level; proceed to search only if confidence < threshold or the answer requires authoritative validation.
3. Source priority: define which sources are primary, which are secondary, and how to resolve conflicts.
4. Query strategy: exact queries to issue, optimized to reduce total calls.
5. Click/open strategy: which results to open and why (minimal but sufficient).
6. Stopping conditions: when to halt searching (e.g., once authoritative source confirms key data).
7. Final reconciliation: how to compare sources, resolve discrepancies, and finalize the answer.

Format requirements:
- Numbered steps, maximum {max_plan_steps}.
- Each step starts with an action verb (e.g., "Disambiguate", "Draft", "Search", "Cross-check", "Reconcile").
{language_requirement}
- Do not include citations, references, or experiment proposals.
- Output only the search plan, nothing else.

Please output the plan in numbered steps like:
(1) ...
(2) ...
etc.

Now generate the plan in up to {max_plan_steps} numbered steps following the above rules."""


# query: OpenAI GPT-4 training dataset size site:openai.com


def get_deep_web_explorer_instruction(
    original_question,
    search_query,
    search_intent,
    search_result,
    language_hint: str = "",
):
    hint = _prefix_language_hint(language_hint)
    return f"""You are a web explorer analyzing search results to find relevant information based on a given search query, search intent and the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the **Original Question**.

2. **More Information Seeking:**
- If the information is not relevant to the query, you could:
  1. Search again: <|begin_search_query|>another search query<|end_search_query|>
  2. Access webpage content using: <|begin_click_link|>your URL<|end_click_link|>

3. **Extract Relevant Information:**
- Return the relevant information from the **Searched Web Pages** that is relevant to the **Current Search Query**.
- Return information as detailed as possible, do not omit any relevant information.
- Make sure the extracted information is useful for answering the **Original Question**.

4. **Output Format:**
- Present the information beginning with **Final Information** as shown below.

**Final Information**
[Relevant information]

**Inputs:**

- **Original Question:**
{original_question}

- **Current Search Query:**
{search_query}

- **Detailed Search Intent:**
{search_intent}

- **Searched Web Pages:**
{search_result}

{hint}Now please analyze the web pages and extract relevant information for the search query "{search_query}" and the search intent, in the context of the original question.
"""


def get_click_web_page_reader_instruction(click_intent, document):
    return f"""Please provide all content related to the following click intent from this document in markdown format.

Click Intent: 
{click_intent}

Searched Web Page:
{document}

Instructions:
- Extract all content that matches the click intent, do not omit any relevant information.
- If no relevant information exists, output "No relevant information"
- Focus on factual, accurate information that directly addresses the click intent
"""


def get_search_intent_instruction(question, prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest search query.
Original question: {question}
Previous thoughts: {prev_reasoning}
Please provide the current search intent."""


def get_click_intent_instruction(question, prev_reasoning):
    return f"""Based on the previous thoughts below, provide the detailed intent of the latest click action.
Original question: {question}
Previous thoughts: {prev_reasoning}
Please provide the current click intent."""


def get_write_section_instruction(
    question,
    previous_thoughts,
    relevant_documents,
    section_name,
    section_goal,
    current_article,
    language: str = "en",
):
    language_note = _language_note(
        language,
        "- 本节内容请以简体中文为主，必要的英文术语或缩写可直接保留，确保段落完整且术语统一。\n",
        "- Write this section in English with complete, well-structured paragraphs.\n",
    )
    return f"""You are a research paper writing assistant. Please write a complete and comprehensive "{section_name}" section based on the following information.

Potential helpful documents (maybe blank):
{relevant_documents}

Original question:
{question}

Previous thoughts:
{previous_thoughts}

Outline of current written article:
{current_article}

Name of the next section to write:
## {section_name}

Your task is to comprehensively write the next section based on the following goal:
{section_goal}

Note:
- Write focused content that aligns with the above goal for this section.
- No need to mention citations or references.
- Each paragraph should be comprehensive and well-developed to thoroughly explore the topic. Avoid very brief paragraphs that lack sufficient detail and depth.
- If possible, add markdown tables to present more complete and structured information to users.
{language_note}

Please provide the comprehensive content of the section in markdown format.
## {section_name}
"""


def get_section_summary_instruction(section):
    return f"""Provide an extremely concise summary of each paragraph or subsection in the following section:
{section}
"""


def get_edit_article_instruction(edit_instruction, article, language: str = "en"):
    language_note = _language_note(
        language,
        "- 最终输出请以简体中文为主，可保留必要的英文术语或缩写。\n",
        "- The final output must present the entire article in English.\n",
    )
    return f"""You are a professional article editor. Please help me modify the article based on the following edit instruction:

Edit instruction:
{edit_instruction}

Current article:
{article}

Please output the complete modified article incorporating all the requested changes.

Note:
- Keep all original content that doesn't need modification. (Do not just output the modified content, but output the entire modified article.)
- Make all edits specified in the edit instructions.
- Output format:
```markdown
...
```
{language_note}

Please provide the complete modified article in markdown format."""


def get_edit_section_instruction(edit_instruction, article):
    return f"""You are a professional article editor. Please help me modify the article based on the following edit instruction:

Edit instruction:
{edit_instruction}

Current article:
{article}

Please first output the entire section/subsection that needs to be modified, then provide the entire modified section/subsection, both in markdown format.

Output Format:

Entire section/subsection to modify:
```markdown
...
```

Entire modified section/subsection:
```markdown
...
```
"""


def get_title_instruction(question, article, language: str = "en"):
    directive = _language_note(
        language,
        "请直接输出以简体中文为主的标题，可保留必要的英文术语或缩写，不要包含任何额外文本。",
        "Directly output the title in English with no additional text.",
    )
    return f"""Please generate a precise title for the following article:

Original Question:
{question}

Currect Article:
{article}

{directive}"""


def get_final_report_instruction(question, article, language: str = "en"):
    language_note = _language_note(
        language,
        "- 输出的最终文章应以简体中文为主，可保留必要的英文术语或缩写。\n",
        "- The final article must be written in English.\n",
    )
    code_stub = _language_note(language, "最终版文章。", "The final-version article.")
    return f"""You are an final-version article editor. Your task is to correct the structure of the following article draft.

Original Question:
{question}

Current Article:
{article}

Note:
- Output the complete final-version article.
- Remove duplicate or redundant content. If there is no error, just output the original article.
- Focus on structure only. Do not omit any valid contents/tables in current article.
{language_note}

Output Format:
```markdown
{code_stub}
```
"""


def get_standard_rag_report_instruction(question, documents):
    return f"""You are a research assistant. Please write a comprehensive research article based on the following question and retrieved documents.

Research Question: {question}

Retrieved documents:
{documents}

Please write a comprehensive research article in markdown format. Do not add citations or references.

Output Format:
```markdown
...
```
"""


def get_direct_gen_report_instruction(question):
    return f"""You are a research assistant. Please write a comprehensive research article based on the following question and answer.

Research Question: {question}

Please write a comprehensive research article in markdown format.

Output Format:
```markdown
...
```
"""


def get_summarization_instruction(question: str, search_query: str) -> str:
    return f"""
    You summarize a webpage TO HELP ANSWER A SPECIFIC QUESTION. Use query-focused summarization.

    Question:
    {question}

    Search query that led to this page:
    {search_query}

    Extract ONLY information directly useful to answer the question. 
    Prefer concrete facts (names, dates, numbers, definitions, lists, step counts, colors, etc.). 
    Avoid background if not needed. Do NOT speculate.

    If the page is irrelevant or lacks evidence for the question:
    - Set "summary" to an empty string "".
    - Still provide at least one brief item in "key_points" explaining irrelevance (schema requires ≥1).

    Output STRICTLY in JSON with this schema and nothing else:
    {{
    "title": "A short title of the page (or null if not available)",
    "summary": "A concise paragraph (4-6 sentences, ~100-150 words) focused on answering the question using page facts; leave empty string if irrelevant",
    "key_points": ["3-5 bullets of the most question-relevant facts; if insufficient, include one bullet explaining irrelevance/insufficiency"]
    }}
    No markdown, no extra text outside JSON, no citations.
    """


def get_detailed_web_page_reader_instruction(
    query, search_intent, document, language_hint: str = ""
):
    hint = _prefix_language_hint(language_hint)
    return f"""Please provide all content related to the following search query and search intent from this document in markdown format.

Search Query: 
{query}

Search Intent: 
{search_intent}

Searched Web Page:
{document}

Instructions:
- Extract all content that matches the search query and intent, do not omit any relevant information.
- Include any relevant links from the source
- If no relevant information exists, output "No relevant information"
- Focus on factual, accurate information that directly addresses the query/intent
{hint}
"""


def get_web_page_reader_instruction(query, document, language_hint: str = ""):
    hint = _prefix_language_hint(language_hint)
    return f"""{hint}{document}
Please provide all content related to "{query}" from this document in markdown format.
If there isn't any relevant information, just output "No relevant information". If there is any relevant information, output all the relevant information with potential helpful links."""


def get_query_plan_instruction(question):
    return f"""You are a reasoning assistant. Your task is to generate a detailed query plan for answering the user's question by breaking it down into sub-queries.

Question: {question}

Please analyze the question and break it down into multiple sub-queries that will help gather all the necessary information to answer it completely. 

Output your query plan in JSON format as follows:

```json
{{
    "query_plan": [
        "sub-query-1",
        "sub-query-2",
        ...
    ]
}}
```
"""


def get_webpage_to_reasonchain_instruction(prev_reasoning, search_query, document):
    return f"""**Task Instruction:**

You are tasked with reading and analyzing web pages based on the following inputs: **Previous Reasoning Steps**, **Current Search Query**, and **Searched Web Pages**. Your objective is to extract relevant and helpful information for **Current Search Query** from the **Searched Web Pages** and seamlessly integrate this information into the **Previous Reasoning Steps** to continue reasoning for the original question.

**Guidelines:**

1. **Analyze the Searched Web Pages:**
- Carefully review the content of each searched web page.
- Identify factual information that is relevant to the **Current Search Query** and can aid in the reasoning process for the original question.

2. **Extract Relevant Information:**
- Select the information from the Searched Web Pages that directly contributes to advancing the **Previous Reasoning Steps**.
- Ensure that the extracted information is accurate and relevant.

3. **Output Format:**
- **If the web pages provide helpful information for current search query:** Present the information beginning with `**Final Information**` as shown below.
**Final Information**

[Helpful information]

- **If the web pages do not provide any helpful information for current search query:** Output the following text.

**Final Information**

No helpful information found.

**Inputs:**
- **Previous Reasoning Steps:**  
{prev_reasoning}

- **Current Search Query:**  
{search_query}

- **Searched Web Pages:**  
{document}

Now you should analyze each web page and find helpful information based on the current search query "{search_query}" and previous reasoning steps.
"""


def get_query_clarification_instruction(question):
    """Adapted from OpenAI's Deep Research guide: https://platform.openai.com/docs/guides/deep-research"""
    return f"""
You are talking to a user who is asking for a research task to be conducted.
The user's question is:
"{question}"

Your job is to gather more information from the user to successfully complete the task.

GUIDELINES:
- Be concise while gathering all necessary information
- Make sure to gather all the information needed to carry out the research task in a concise, well-structured manner.
- Use bullet points or numbered lists if appropriate for clarity.
- Don't ask for unnecessary information, or information that the user has already provided.

IMPORTANT: Do NOT conduct any research yourself, just gather information that will be given to a researcher to conduct the research task.
"""


def get_query_rewriting_instruction(question, added_info):
    return f"""
You will be given a research task by a user.

Original user query:
"{question}"

Additional info after clarification:
"{added_info}"

Your job is to produce a set of
instructions for a researcher that will complete the task. Do NOT complete the
task yourself, just provide instructions on how to complete it.

GUIDELINES:
1. **Maximize Specificity and Detail**
- Include all known user preferences and explicitly list key attributes or
  dimensions to consider.
- It is of utmost importance that all details from the user are included in
  the instructions.
- **Base your instructions on BOTH the Original user query and the Additional info above.**

2. **Fill in Unstated But Necessary Dimensions as Open-Ended**
- If certain attributes are essential for a meaningful output but the user
  has not provided them, explicitly state that they are open-ended or default
  to no specific constraint.

3. **Avoid Unwarranted Assumptions**
- If the user has not provided a particular detail, do not invent one.
- Instead, state the lack of specification and guide the researcher to treat
  it as flexible or accept all possible options.

4. **Use the First Person**
- Phrase the request from the perspective of the user.

5. **Tables**
- If you determine that including a table will help illustrate, organize, or
  enhance the information in the research output, you must explicitly request
  that the researcher provide them.

Examples:
- Product Comparison (Consumer): When comparing different smartphone models,
  request a table listing each model's features, price, and consumer ratings
  side-by-side.
- Project Tracking (Work): When outlining project deliverables, create a table
  showing tasks, deadlines, responsible team members, and status updates.
- Budget Planning (Consumer): When creating a personal or household budget,
  request a table detailing income sources, monthly expenses, and savings goals.
- Competitor Analysis (Work): When evaluating competitor products, request a
  table with key metrics, such as market share, pricing, and main differentiators.

6. **Headers and Formatting**
- You should include the expected output format in the prompt.
- If the user is asking for content that would be best returned in a
  structured format (e.g. a report, plan, etc.), ask the researcher to format
  as a report with the appropriate headers and formatting that ensures clarity
  and structure.

7. **Language**
- If the user input is in a language other than English, tell the researcher
  to respond in this language, unless the user query explicitly asks for the
  response in a different language.

8. **Sources**
- If specific sources should be prioritized, specify them in the prompt.
- For product and travel research, prefer linking directly to official or
  primary websites (e.g., official brand sites, manufacturer pages, or
  reputable e-commerce platforms like Amazon for user reviews) rather than
  aggregator sites or SEO-heavy blogs.
- For academic or scientific queries, prefer linking directly to the original
  paper or official journal publication rather than survey papers or secondary
  summaries.
- If the query is in a specific language, prioritize sources published in that
  language.
"""
