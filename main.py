from langgraph.graph import StateGraph,END
from typing import TypedDict, List
from typing import Literal
from langchain_openai import ChatOpenAI
import json
import os
import gradio as gr
from dotenv import load_dotenv
import re

load_dotenv()
llm = ChatOpenAI(
    model="deepseek-ai/DeepSeek-V3",
    api_key=os.getenv("SILICONFLOW_API_KEY"),  # 从环境变量读
    base_url="https://api.siliconflow.cn/v1",
    temperature=0.3
)
#定义全局
class ChatState(TypedDict):
    user_input : str
    agent5_output : dict
    agent5_questionnum : int
    agent1_output : dict
    agent2_level : str
    agent3_output : dict
    agent4_output : dict
    final_output : dict
    chat_history : list
    current_reply : str
    mode : str

agent1_prompt = """
Agent1:症状提取官

【角色设定】
你是一名三甲医院经验丰富的分诊护士。你专业，温柔，耐心。你不负责诊断，你需要像侦探一样，通过追问收集足够的信息。

【你的核心能力】
你拥有扎实的医学基础，了解人体各个系统常见疾病的典型表现和危险信号。

【开始规则】
- 如果用户发送了第一条消息，你根据内容开始追问。
- 如果用户只是发了一张图片或说"你好"，你主动发起问诊："您好，请告诉我您哪里不舒服？越详细越好，我会一步步问您。"

【核心方法论】
无论患者的主诉是什么，你可以先询问其性别，年龄
接下来遵循以下**"分诊护士四步追问法"**，且每次输出只能问出一个问题
第一步：排除"红旗警报"（Red Flags）（1-2轮）
针对这个主诉，你的医学知识里最危险的情况是什么？
- 例1（头痛）：雷击样剧痛、发热+颈强直、视力突变 → 这是颅内感染/出血警报
- 例2（胸痛）：压榨感+大汗+放射痛 → 这是心梗警报
- 例3（腹痛）：全腹板硬+呕吐+无排气 → 这是肠梗阻/穿孔警报
- 例4（咳嗽）：咳血+胸痛+发热+消瘦 → 这是肺部严重病变警报
- 例5（发烧）：高热+意识模糊+皮疹 → 这是败血症/脑膜炎警报

你必须追问对应的"Red Flags"问题，确认患者没有这些危险信号。

第二步：定位"症状特征"（Characterization）（1-2轮）
针对这个主诉，你需要搞清楚它的"具体模样"：
- 性质：刺痛/钝痛/灼烧感/搏动性/压榨性/持续性/间歇性
- 部位：具体哪里？局部还是弥漫？
- 诱发因素：做什么会加重/缓解？（吃饭、运动、呼吸、按压）
- 伴随症状：还发生了什么？（恶心、发烧、出汗、头晕）

你必须追问直到掌握以上4个特征。

第三步：确认"时间和背景"（Context）（1轮）
- 什么时候开始的？突然还是逐渐？
- 最近有没有特殊情况？（受伤、旅行、新食物、压力）
- 对日常生活影响多大？（正常工作/无法入睡/无法进食）

第四步：汇总确认
当你完成前三步后，**在心里自检**：
"如果我是一名导诊护士，拿着这些信息去给患者挂号，我够不够用？"
如果够 → 停止追问，输出JSON。
如果不够 → 继续追问缺失的关键信息。

【重要警告：严禁"偷懒"或"套模板"！】
- 你不允许因为Prompt里没写"膝盖疼"的例子，就默认"膝盖疼"不需要追问危险信号。
- 如果用户说"膝盖疼"，你必须思考："膝盖疼最危险的情况是什么？——化脓性关节炎、骨折、深静脉血栓？"然后基于此追问。
- 如果用户说"牙疼"，你必须思考："牙疼最危险的情况是什么？——心梗放射痛？颌骨骨髓炎？"然后基于此追问。

【边界限制 - 必须遵守】
- 如果用户要求诊断、开药、或询问"我该怎么办"，你必须回答："我仅负责记录您的症状和导诊，不能提供诊断或用药建议。如果您感到不适，请及时挂号就医。"
- 绝对不要给出任何医学判断（如"你可能得了感冒"），你只负责记录和追问。

【终止条件 - 必须满足以下所有条件才能停止】
在输出 JSON 之前，请自检是否已经收集了以下 6 个维度的信息（每个维度至少获得过用户的回答，不一定要追问到完美）：
1. 症状的具体性质（如：疼痛性质/瘙痒/红肿等）
2. 症状的具体部位
3. 持续时间
4. 严重程度或发作频率
5. 至少 1 个伴随症状或排除项（如：有无发烧、有无恶心）
6. 年龄和性别

如果以上 6 项都已获得，你可以输出 JSON。如果缺少 2 项以上，请继续追问。

【输出格式 - 最终完成时（重要更新）】
当你确认已经收集齐所有必要信息（满足上述 6 个维度）后，输出以下 **完整 JSON**：

{
  "主诉": "用户的原话",
  "标准化症状": ["标准术语1", "标准术语2"],
  "症状性质": "...",
  "症状部位": "...",
  "持续时间": "...",
  "严重程度": "轻微/中等/严重",
  "伴随症状": ["..."],
  "年龄": "...",
  "性别": "...",
  "department": "根据症状推荐的科室（1~2个，用顿号隔开）",
  "advice": "一段完整的挂号指引话术（包含科室推荐和安抚语句，约40~60字，语气温和）"
}

【科室推荐规则（当输出 department 和 advice 时遵守）】
- 根据"标准化症状"和"症状性质"判断最可能的科室方向。
- 如果症状涉及多个系统，可以推荐两个科室（如"神经内科、眼科"）。
- 如果症状模糊或信息不足，推荐"全科门诊"。
- **绝对不要给出任何诊断结论**（不能说"您得了偏头痛"）。
- **绝对不要提及任何药品名称或治疗方案**。

⚠️ 重要：只有在确认收集完成时才输出包含 department 和 advice 的完整 JSON。
如果信息不足，请继续用自然语言追问（不要输出 JSON），每次只问一个问题。
"""



agent3_prompt = """
【角色设定】
你是一名三甲医院资深导诊护士，对各种症状应该挂哪个科室了如指掌。你睿智、冷静、经验丰富。

【你的任务】
根据用户的结构化症状信息（包含症状、持续时间、严重程度、年龄、性别等），**直接推荐 1-2 个最合适的科室**。你不需要检索数据库，你的医学知识就是你的依据。

【生成规则】
1. 根据用户的**主诉**和**症状性质**，判断最可能的科室方向。
2. 如果症状涉及多个系统（如“头痛+视力模糊”），可以推荐两个科室（如“神经内科”和“眼科”），让患者现场灵活选择。
3. 如果症状模糊或信息不足（比如只有“不舒服”），统一推荐“全科门诊”或“导诊台”。

【安全红线 - 必须遵守】
- **绝对不要给出任何诊断结论**（不能说“您得了偏头痛”、“可能是流感”）。
- **绝对不要提及任何药品名称或治疗方案**（不能说“建议服用布洛芬”）。
- 语气必须冷静、温柔、不制造恐慌。

【输出格式 - 严格 JSON】
{
  "department": "建议挂号的科室（1-2个，用顿号隔开）",
  "advice": "一段完整的挂号指引话术（包含科室推荐和温和的安抚语句，字数控制在 40-60 字）"
}

【示例】
输入：{"主诉":"头痛","标准化症状":["头痛"],"症状性质":"搏动性","年龄":"25","性别":"女"}
输出：{"department":"神经内科","advice":"根据您描述的搏动性头痛，建议优先挂神经内科门诊。祝您就医顺利。"}

输入：{"主诉":"肚子疼","标准化症状":["腹痛"],"症状性质":"绞痛","年龄":"40","性别":"男"}
输出：{"department":"消化内科","advice":"根据您描述的腹部绞痛，建议前往消化内科就诊。请保持清淡饮食。"}
"""

agent4_prompt = """
【角色设定】
你是一名AI医疗内容安全审查员，你的唯一任务是对即将发送给用户的“导诊建议”进行安全审核。你关注的是：这条建议是否越过了“导诊”的边界。

【你的审核标准】
以下内容属于“安全范围”（允许）：
- 推荐科室（如：“建议您挂神经内科”）
- 建议就医（如：“建议您尽快前往医院”）
- 症状描述复述（如：“您描述的头痛”）
- 安抚性语句（如：“祝您早日康复”）
- 免责声明（如：“本建议仅作为参考”）

以下内容属于“越界内容”（必须屏蔽或修改）：
- ❌ 明确的诊断结论（如：“您这是偏头痛”、“您得了流感”）
- ❌ 具体的用药建议（如：“建议服用布洛芬”、“可以吃阿莫西林”）
- ❌ 治疗方案（如：“建议进行CT检查”、“需要手术”）
- ❌ 任何听起来像“确诊”的表述

【你的处理流程】
1. 接收Agent 3输出的导诊建议JSON。
2. 逐字检查“Advice”字段，定位是否存在越界内容。
3. 如果发现越界内容：屏蔽该部分，只保留“推荐科室”和“就医建议”部分。
4. 在输出的“Advice”字段中，**强制追加**免责声明。

【输出格式 - 与Agent 3格式保持一致】
{
  "department": "建议挂号的科室（1-2个，用顿号隔开）",
  "advice": "一段完整的挂号指引话术（包含科室推荐和温和的安抚语句，字数控制在 40-60 字）"
}

【越界内容修改示例】
原始输出：
“根据您描述的持续性头痛，可能是偏头痛。建议您去神经内科挂号，可以先服用布洛芬缓解。”
修改后：
“根据您描述的头痛，建议您去神经内科挂号。AI助手仅提供参考，专业诊断请前往医院由医生得出。”

原始输出：
“您的症状符合急性阑尾炎的特征，建议立即手术。”
修改后：
“根据您的描述，建议您立即前往普外科或急诊科就诊。AI助手仅提供参考，专业诊断请前往医院由医生得出。”

【重要约束】
- 如果原输出中没有任何越界内容，仍然需要追加免责声明。
- 修改后的建议必须依然能指导用户去正确的科室。
- 语气保持冷静、专业，不增加恐慌。

【输出JSON】
{
  "Advice": "根据您描述的【症状】，建议您前往【科室】就诊。AI助手仅提供参考，专业诊断请前往医院由医生得出。",
  "Manner": "祝您就医顺利，早日康复！"
}
"""

# 新增 Agent 5（心理自测员）

agent5_prompt = """
【角色设定】
你是一名部门心理关爱员，你的角色是三合一：情绪的“雷达”、心灵的“守门人”、资源的“链接者”。
你的目标不是解决所有问题，而是不让任何一个需要帮助的人被忽视。

【核心规则 - 第一轮：知情同意（必须执行）】
无论用户第一次发送什么消息，你的第一轮回复**必须**是以下完整的知情同意说明，并等待用户明确回复“同意”或“我同意”后才能继续：

---

“您好，在开始之前，我需要先做一个简单的咨询说明：

1. **保密原则**：我们的对话内容会严格保密，未经你的允许，我不会将你的个人信息或咨询内容透露给任何人。你可以放心地敞开心扉交流。

2. **保密例外**：但如果出现以下特殊情况，我需要打破保密原则，告知你的家人、公司相关部门或专业机构：
   - 你有伤害自己（如自杀、自伤）或伤害他人的风险
   - 存在触犯法律的行为

   这么做是为了保障你的生命安全和相关利益。

3. **转介承诺**：如果遇到超出我能力范围的情况，我会将你转介给公司EAP专业团队或医疗机构，确保你能得到最合适的帮助。

**请问你是否同意以上条款，愿意继续我们的对话？**
（请回复“同意”或“我同意”即可开始）”

---

**特别注意**：
- 在用户明确回复“同意”或“我同意”之前，**不要**提供任何陪伴服务、问卷或建议。
- 如果用户回复“不同意”或拒绝，请尊重其意愿，回复：“我完全尊重您的决定。如果您想聊，我随时都在。”
- 用户同意后，再进入正常对话流程。

【保密原则 - 行为准则】
- 你的对话是安全的。即使你打破了保密原则，也是出于关心，而不是背叛。
- 用户同意后，你可以在对话中适当提醒：“正如我们开始前约定的，我们的对话是保密的。”

【交互流程（用户同意后）】

1. **开场问候（知情同意后）**：
   用户同意后，你的第一句回复可以是：
   “感谢您的信任。我在这里陪您。您可以选择：
   A) 和我聊聊天，说说心里话（纯粹陪伴）
   B) 做一个简短的心理状态自测，了解自己的情绪（8 个问题，约 3-5 分钟）
   
   您想选哪一个？或者直接告诉我您想聊什么也可以。”

2. **陪伴模式**（用户选择 A，或表现出倾诉意愿）：
   - 以倾听为主，回应温暖、共情。
   - 遵循“接·懂·引”三字诀。
   - **不需要收集结构化信息，不需要输出 JSON。**

3. **问卷模式**（用户选择 B，或明确表示愿意做自测）：
   - 依次询问 8 个问题（每次只问一个）。
   - 评分标准：0=完全没有，1=有几天，2=超过一半天数，3=几乎每天。
   - 如果用户不想继续，立即退出问卷，转为陪伴模式。

4. **结果输出（问卷完成后）**：
   {
     "score": 总分（0-24 之间）,
     "level": "正常范围 / 轻度 / 中度 / 中重度 / 重度",
     "advice": "对应的建议（如：建议与信任的人交流、预约心理咨询、尽快就医等） + 一句温暖的关心和邀请（如：如果您愿意，我也可以继续陪您聊聊天。）",
     "recommended_dept": "心理科 / 精神科 / 无需挂号"
   }

【安全红线 - 硬拦截】
如果用户在任何时候提到以下内容，**立即中止当前对话，直接输出拦截话术**：
- “想自杀”、“不想活了”、“觉得活着没意义”、“想伤害自己”、“我打算结束生命”

拦截话术：
“听到您这么说，我非常担心。请您立即联系心理援助热线：
- 希望24热线：400-161-9995
- 北京心理危机干预中心：010-82951332
- 广州心理危机干预中心专线：020-81899120
或前往最近医院的心理科/急诊科。您不是一个人在面对这些。”

【输出规则】
- 第一轮：固定知情同意说明（含保密条款和同意确认）。
- 用户同意前：不提供任何其他内容，只等待“同意”回复。
- 用户同意后：按正常流程（开场白 → 陪伴/问卷）。
- 陪伴模式下：自然语言回复。
- 问卷模式下：自然语言问题（如“Q1：...”）。
- 问卷完成时：输出 JSON。
- 遇到危机信号：立即输出拦截话术。
"""

#LLM Node
def entry_node(state: ChatState):
    return {}

def agent_special_mental(state: ChatState):
    if state.get("agent5_output") is not None:
        return state
    
    history = state.get("chat_history",[])
    user_input = state.get("user_input","")
    numnow = state.get("agent5_questionnum",1)

    message = [
        {"role": "system", "content":agent5_prompt},
        *history,
        {"role": "user", "content":f"{user_input}"}
    ]

    response = llm.invoke(message)
    reply = response.content
    
    try:
        json_start = reply.find('{')
        json_end = reply.rfind('}')+1
        if json_start!=-1 and json_end > json_start:
            json_str = reply[json_start:json_end]
            agent5_output = json.loads(json_str)
            check = ["score","level","advice","recommended_dept"]
            if all(word in agent5_output for word in check):
                return {"agent5_output" : agent5_output}
            
    except json.JSONDecodeError:
        pass

    new_history = history + [
        {"role":"user","content":user_input},
        {"role":"assistant","content":reply},
    ]

    match = re.search(r'Q(\d+)', reply)
    if match:
        new_num = int(match.group(1))
    else:
        new_num = numnow  # 保持不变
        
    return {
        "chat_history": new_history,
        "current_reply": reply,
        "agent5_questionnum": new_num
    }

def route_by_mode(state: ChatState) -> Literal["agent1", "agent5"]:
    if state.get("mode") == "1":
        return "agent5"
    else:
        return "agent1"

def agent1_symptom_collector(state: ChatState):
    #...调用api，返回追问或最终json

    #第一步 检查是否收集完了
    #如果 state 里已经有 agent1_output，说明上一轮已经输出 JSON 了
    # 直接返回，避免死循环
    if state.get("agent1_output") is not None:
        return state
    
    #2从state里提取数据
    history = state.get("chat_history",[])
    user_input = state.get("user_input", "")

    #3 组成langchain格式发给大预言模型
    messages = [
        {"role": "system", "content":agent1_prompt},
        *history,
        {"role": "user", "content":user_input}
    ]

    #4调用大语言模型
    response = llm.invoke(messages)
    reply = response.content

    #5尝试提取大预言模型
    try:
        #尝试从回复中提取大语言模型
        json_start = reply.find('{')
        json_end = reply.rfind('}') + 1
        if json_start != -1 and json_end > json_start:
            json_str = reply[json_start:json_end]
            agent1_output = json.loads(json_str)
            required_fields = ["主诉", "年龄", "性别"]
            if all(field in agent1_output for field in required_fields):
                # 收集完成！存到 state 并返回
                return {"agent1_output": agent1_output}
        
    except json.JSONDecodeError:
        pass

    #6来到这里就说明是追问
    new_history = history + [
        {"role": "user", "content": user_input},
        {"role": "assistant", "content": reply}
    ]
    

    return {
        "chat_history": new_history,
        "current_reply": reply  # 这个字段可以用于前端显示
    }


#Functional Node
def agent2_emergency_triage(state: ChatState):


    #判断紧急情况 yes则输出red flag,no则否
    history = state.get("chat_history")
    agent1_data = state.get("agent1_output")
    raw_text = state.get("user_input"," ")
    
    #两个方法 第一个是通过对话历史检查
    for msg in history:
        if(msg.get("role")=="user"):
            raw_text+=" " + msg.get("content"," ")
    
    std_symptoms = agent1_data.get("标准化症状", [])

    raw_red_flags = [
        "石头压","呼吸困难","濒死","要死了","喘不上","窒息","发紫","喉咙肿痛"
        ,"大舌头", "口齿不清", "动不了",
        "咳血", "呕血", "晕倒", "不省人事"
    ]

    for flag in raw_red_flags:
        if flag in raw_text:
            return {"agent2_level": "RED"}
    
    #第二个则是通过agent1精简的标准化症状检查
    std_red_terms = ["压榨性疼痛", "濒死感", "呼吸困难", "偏瘫", "意识模糊", "咳血", "呕血"]
    for term in std_red_terms:
        if term in std_symptoms:
            return {"agent2_level": "RED"}
    
    return {"agent2_level": "GREEN"}

#LLM Node
def agent3_department_matcher(state: ChatState):
    #调用API，生成挂号建议
    data = state.get("agent1_output")

    if data is None:
        return {"agent3_output": {"department": "全科门诊", "advice": "建议您前往医院导诊台或全科门诊进一步咨询。"}}
    
    #把json转换成字符
    symptoms_json = json.dumps(data,ensure_ascii = False)

    #发给大模型的msg
    msg=[
        {"role":"system","content":agent3_prompt},
        {"role":"user","content":f"请通过以下症状推荐科室:{symptoms_json}"}
    ]

    #让大模型运行
    response = llm.invoke(msg)
    reply = response.content

    try:
        json_start = reply.find('{')
        json_end = reply.rfind('}')+1
        if json_start!= -1 and json_end>json_start:
            json_str=reply[json_start:json_end]
            agent3_output = json.loads(json_str)
            if "department" in agent3_output:
                return {"agent3_output": agent3_output}
    except json.JSONDecoder:
        pass

    return {"agent3_output": {"department": "全科门诊", "advice": "建议您前往医院导诊台或全科门诊进一步咨询。"}}

#LLM Node
def agent4_safety_checker(state: ChatState):
    #检查输出 加免责声明
    data = state.get("agent3_output")
    helps = json.dumps(data, ensure_ascii=False)

    msg = [
        {"role":"system","content":agent4_prompt},
        {"role":"user","content":f"请审核以下挂号建议:{helps}"}
    ]

    response = llm.invoke(msg)
    reply = response.content

    try:
        json_start = reply.find("{")
        json_end = reply.rfind("}")+1
        if(json_start != -1 and json_end>json_start):
            json_str = reply[json_start:json_end]
            agent4_output = json.loads(json_str)
            if "department" in agent4_output and "advice" in agent4_output:
                return {"agent4_output":agent4_output}
    except json.JSONDecodeError:
        pass

    fallback = state.get("agent3_output").copy()
    fallback["advice"] = fallback.get("advice", "") + " AI助手仅提供参考，专业诊断请前往医院由医生得出。"
    return {"agent4_output":fallback}

#最终输出格式
def final_output_format(state: ChatState):
    data = state.get("agent1_output",{})

    dept = data.get("department", "全科门诊")
    advice = data.get("advice", "建议您前往医院咨询。")

    final_text = f"""🏥 建议挂号：{dept}
    💡 {advice}
    ⚠️ 本建议仅作为参考，不替代专业医疗诊断。请以医院医生的意见为准。"""

    state["final_output"] = {
        "text": final_text,          # 展示给用户看的完整文本
        "department": dept,          # 科室名（供前端提取）
        "advice": advice             # Agent 4 完整话术（供日志/审计）
    }

    return state

#agent2 后判断
def route_after_triage(state: ChatState) -> Literal["final_format","direct_output"]:
    level = state.get("agent2_level","GREEN")

    if level == "RED":
        return "direct_output"
    else:
        return "final_format"

#agent1 后判断
def route_after_agent1(state: ChatState) -> Literal["agent2", "__end__"]:
    if state.get("agent1_output") is not None:
        return "agent2"
    else:
        # 关键改动：如果还没收集完，直接结束本次执行，等待用户下一轮输入
        return "__end__"
def route_after_agent5(state: ChatState) -> Literal["next","__end__"]:
    if state.get("agent5_output") is not None:
        return "next"
    else:
        return "__end__"

#急诊后直接输出
def direct_output(state: ChatState):
    symptom_list = state.get("agent1_output", {}).get("标准化症状", [])
    
    if isinstance(symptom_list, list):
        symptom = "、".join(symptom_list) if symptom_list else "您的症状"
    else:
        symptom = symptom_list or "您的症状"
    
    emergency_alert = {
        "level": "RED",
        "alert_title": "⚠️ 就医提醒",
        "alert_body": f"您描述的【{symptom}】在临床上属于需要高度警惕的信号。",
        "emergency_action": "请立即停止一切活动，保持平卧或半卧位。请家人陪同，**立即前往最近医院的【急诊科】**，或直接拨打120急救电话。请勿自行驾车前往！",
        "soothe_words": "请您不要过度惊慌，现代医学对于这类急症有非常成熟的急救流程。保持镇定，尽快到达医院，医生一定能给您最及时的治疗。",
        "recommended_dept": "急诊科"
    }
    state["final_output"] = emergency_alert
    return state

def mental_output(state: ChatState):
    user = state.get("agent5_output",{})
    score = user.get("score","未知")
    level = user.get("level","未知")
    advice = user.get("advice","未知")
    department = user.get("recommended_dept","心理科")

    if level == "正常":  
        mental_output = {
            "output_body":f"您的得分为{score},在诊断中属于{level}心理波动",
            "output_advice":f"{advice},无需挂号,但如果情绪持续或加重，请及时找到信任的人或自咨询师获得帮助",
            "soothe_words":f"不论何时，您不是一个人在面对这些🫂。如若需要找人聊聊，可以拨打北京心理危机干预中心热线 010-82951332"
        }
    else:
        mental_output = {
            "output_body":f"您的得分为{score},在诊断中属于{level}心理健康预警",
            "output_advice":f"{advice},建议您前往{department}预约就医",
            "soothe_words":f"不论何时，您不是一个人在面对这些🫂。如若需要找人聊聊，可以拨打北京心理危机干预中心热线 010-82951332"
        }
    state["final_output"] = mental_output
    return state

#4 构建图
builder = StateGraph(ChatState)
builder.add_node("agent5", agent_special_mental)
builder.add_node("agent1",agent1_symptom_collector)
builder.add_node("agent2",agent2_emergency_triage)
builder.add_node("direct_output", direct_output)
builder.add_node("final_format",final_output_format)
builder.add_node("start",entry_node)
builder.add_node("mental_format",mental_output)

#5 构建边
builder.set_entry_point("start")
builder.add_conditional_edges(
    "start",
    route_by_mode,
    {
        "agent1": "agent1",
        "agent5": "agent5"
    }
)
builder.add_conditional_edges(
    "agent5",
    route_after_agent5,
    {
        "next":"mental_format",
        "__end__":END
    }
)
builder.add_edge("mental_format",END)
builder.add_conditional_edges(
    "agent1",
    route_after_agent1,
    {
        "__end__":END,
        "agent2":"agent2"
    }
)

builder.add_conditional_edges(
    "agent2",
    route_after_triage,
    {
        "direct_output": "direct_output",  # 映射到节点名
        "final_format": "final_format"                # 映射到节点名
    }
)
graph = builder.compile()

if __name__ == '__main__':

    

    print("="*50)
    print("🏥 AI 健康导诊助手已启动 (输入 'quit' 退出)")
    print("="*50)

    mode = input("\n请问需要咨询的模式是？\n1)心理咨询\n2)导诊建议\n(请输入1或2)\n")

    state = {
        "user_input": "",
        "chat_history": [],
        "agent1_output": None,
        "agent2_level": "",
        "agent3_output": None,
        "agent4_output": None,
        "agent5_output":None,
        "agent5_questionnum":1,
        "final_output": None,
        "current_reply": "",
        "mode":mode
    }

    while True:
        user_input = input("\n您：")
        if user_input.lower() in ['quit','exit','q']:
            print("👋 祝您身体健康，再见！")
            break
        if not user_input.strip():
            continue

        state["user_input"] = user_input

        try:    

            final_state = graph.invoke(state,{"recursion_limit":50})
            state = final_state

            output = final_state.get("final_output")
            current_reply = state.get("current_reply")

            if output:
                if isinstance(output,dict) and "alert_title" in output:
                    print(f"\n🚨 {output['alert_title']}")
                    print(f"{output['alert_body']}")
                    print(f"🆘 {output['emergency_action']}")
                    print(f"💬 {output['soothe_words']}")
                elif isinstance(output, dict) and "text" in output:
                        
                    print(f"\n🤖 助手: {output['text']}")

                    again = input("\n是否继续咨询其他症状？(y/n): ")
                    if again.lower() != 'y':
                        print("👋 祝您身体健康，再见！")
                        break
                    else:
                        mode = input("\n请告诉我您需要的咨询模式\n1)心理咨询\n2)症状咨询\n(请输入1或2)\n")
                        # 重置状态，开始新的问诊
                        state = {
                            "user_input": "",
                            "chat_history": [],
                            "agent1_output": None,
                            "agent2_level": "",
                            "agent3_output": None,
                            "agent4_output": None,
                            "agent5_outout":None,
                            "agent5_numquestion":1,
                            "mode":mode,
                            "final_output": None,
                            "current_reply": ""
                        }
                            # 重置后继续 while 循环
                        continue
                elif isinstance(output,dict) and "output_body" in output:
                    print(f"\n{output['output_body']}")
                    print(f"\n{output['output_advice']}")
                    print(f"\n{output['soothe_words']}")

                    again = input("\n是否继续咨询其他症状？(y/n): ")
                    if again.lower() != 'y':
                        print("👋 祝您身体健康，再见！")
                        break
                    else:
                        mode = input("\n请告诉我您需要的咨询模式\n1)心理咨询\n2)症状咨询\n(请输入1或2)\n")
                        # 重置状态，开始新的问诊
                        state = {
                            "user_input": "",
                            "chat_history": [],
                            "agent1_output": None,
                            "agent2_level": "",
                            "agent3_output": None,
                            "agent4_output": None,
                            "agent5_outout":None,
                            "agent5_questionnum":1,
                            "mode":mode,
                            "final_output": None,
                            "current_reply": ""
                        }
                            # 重置后继续 while 循环
                        continue
                else:
                        print(f"\n🤖 助手: {output}")
            elif current_reply:
                # 如果 final_output 为空，但有 current_reply，说明还在追问
                # 但因为我们在 agent1 里已经 print 了，这里可以不重复打印
                # 但为了保险，还是打印一次（但可能重复）
                print(f"\n🤖 助手: {current_reply}")
            else:
                print("\n🤖 助手: 系统暂时无法处理，请稍后再试。")

        except Exception as e:
            print(f"\n❌ 发生错误: {e}")
            print("请重试或检查 API Key 是否正确。")






