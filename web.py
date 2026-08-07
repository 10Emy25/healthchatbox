import gradio as gr
import base64
from main import graph, ChatState


custom_css = """
    /* 整体背景渐变 */
    .gradio-container {
        background: linear-gradient(145deg, #fdf6f0 0%, #f0e6de 100%);
        font-family: 'Georgia', serif;
    }
    /* 聊天气泡圆润化 */
    .message {
        border-radius: 20px !important;
        padding: 15px 20px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    }
    /* 用户气泡颜色（温柔奶茶色） */
    .user .message {
        background-color: #d4c5b2 !important;
        color: #3d2c1e !important;
    }
    /* 机器人气泡颜色（奶白色） */
    .bot .message {
        background-color: #ffffff !important;
        color: #4a3728 !important;
        border: 1px solid #e8ddd0 !important;
    }
    /* 输入框圆角 */
    textarea, .input-box {
        border-radius: 30px !important;
        border: 1px solid #dccfc3 !important;
        background-color: #ffffff !important;
    }
    /* 按钮柔和 */
    .primary {
        background-color: #b8a08c !important;
        color: white !important;
        border-radius: 30px !important;
        border: none !important;
    }
    .primary:hover {
        background-color: #a0856e !important;
    }
"""

    # ... 你的原有代码 ...
# ==================== 读取本地背景图（转为 Base64） ====================
def get_img_base64(path):
    with open(path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

# 读取图片（请确保 bg.jpg 和 web.py 在同一目录）
img_b64 = get_img_base64("bg.jpg")
print(f"✅ 图片已加载，Base64 长度: {len(img_b64)}") 

# 判断图片类型（自动检测）
if "bg.jpg" in "bg.jpg":
    img_type = "jpeg"
elif "bg.png" in "bg.png":
    img_type = "png"
else:
    img_type = "jpeg"  # 默认

# ==================== 核心回复处理 ====================
def respond(message, history, mode):
    """
    message: 用户当前输入（字符串）
    history: 对话历史（列表 of dict，含 role 和 content）
    mode: "导诊建议" 或 "心理健康员"
    返回 (new_history, "") ，new_history 为更新后的历史
    """
    # 1. 转换为 LangGraph 格式
    chat_history = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if content:
            chat_history.append({"role": role, "content": content})

    # 2. 构建状态
    state = {
        "user_input": message,
        "chat_history": chat_history,
        "agent1_output": None,
        "agent2_level": "",
        "agent3_output": None,
        "agent4_output": None,
        "agent5_output": None,
        "agent5_questionnum": 1,
        "final_output": None,
        "current_reply": "",
        "mode": "1" if mode == "心理健康员" else "2"
    }

    # 3. 调用 LangGraph
    try:
        final_state = graph.invoke(state, {"recursion_limit": 50})
        output = final_state.get("final_output")
        current_reply = final_state.get("current_reply")

        # 4. 格式化输出
        reply_text = ""
        if output and isinstance(output, dict):
            if "output_body" in output:          # 心理
                reply_text = (
                    f"{output['output_body']}\n\n"
                    f"{output.get('output_advice', '')}\n\n"
                    f"{output.get('soothe_words', '')}"
                )
            elif "text" in output:               # 导诊
                reply_text = output["text"]
            elif "alert_title" in output:        # 急诊
                reply_text = (
                    f"🚨 {output['alert_title']}\n\n"
                    f"{output['alert_body']}\n\n"
                    f"🆘 {output['emergency_action']}\n\n"
                    f"💬 {output['soothe_words']}"
                )
            else:
                reply_text = str(output)
        elif current_reply:
            reply_text = current_reply
        else:
            reply_text = "系统暂时无法处理，请稍后再试。"

        # 5. 更新历史
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply_text})
        return history, ""

    except Exception as e:
        import traceback
        print("❌ 错误详情:", traceback.format_exc())
        error_msg = f"❌ 发生错误: {e}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""

# ==================== 界面回调函数 ====================

# 发送消息
def send_message(message, history, mode):
    if not message.strip():
        return history, history, ""
    new_history, _ = respond(message, history, mode)
    return new_history, new_history, ""

# 清空
def clear_history():
    return [], [], ""

# 切换模式
def switch_mode(new_mode):
    return [], [], "", new_mode

# ==================== 构建 Gradio 界面 ====================
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="stone")) as demo:
    # ========== 背景图片（使用 Base64 注入） ==========
    gr.HTML(f"""
    <style>
        /* 1. 设置页面背景图（覆盖整个页面） */
        body {{
            background-image: url('data:image/{img_type};base64,{img_b64}') !important;
            background-size: cover !important;
            background-position: center !important;
            background-attachment: fixed !important;
            background-repeat: no-repeat !important;
        }}
        
        /* 2. 让 Gradio 主容器透明，露出 body 背景 */
        .gradio-container {{
            background: rgba(255, 255, 255, 0.88) !important;  /* 半透明白色，保证文字清晰 */
            max-width: 100% !important;
            margin: 0 !important;
            border-radius: 0 !important;
            box-shadow: none !important;
        }}
        
        /* 3. 其他内部组件透明 */
        .gradio-app, .chatbot, .message {{
            background: transparent !important;
        }}
        
        /* 4. 让聊天框本身半透明更柔和 */
        .chatbot {{
            background: rgba(255, 255, 255, 0.5) !important;
        }}
    </style>
    """)

    # ========== 背景音乐（轻柔钢琴曲，使用可靠直链） ==========

    # ========== 标题 ==========
    gr.Markdown("""
    # 🏥 AI 健康导诊助手
    ### 选择模式后开始对话，切换模式会自动清空历史
    """)

    # ========== 工具栏 ==========
    with gr.Row():
        mode_radio = gr.Radio(
            choices=["导诊建议", "心理健康员"],
            value="导诊建议",
            label="服务模式",
            interactive=True
        )
        clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")

    # ========== 聊天显示区 ==========
    chatbot = gr.Chatbot(label="对话", height=400)
    history_state = gr.State([])   # 存储对话历史

    # ========== 输入区域 ==========
    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="请输入您的症状或情绪困扰...",
            scale=4,
            show_label=False,
            lines=2
        )
        send_btn = gr.Button("发送", variant="primary", scale=1)

    # ========== 事件绑定 ==========
    # 文本发送（按钮 + 回车）
    send_btn.click(
        send_message,
        inputs=[msg_box, history_state, mode_radio],
        outputs=[chatbot, history_state, msg_box]
    )
    msg_box.submit(
        send_message,
        inputs=[msg_box, history_state, mode_radio],
        outputs=[chatbot, history_state, msg_box]
    )

    # 清空对话
    clear_btn.click(
        clear_history,
        outputs=[chatbot, history_state, msg_box]
    )

    # 模式切换 → 清空历史
    mode_radio.change(
        switch_mode,
        inputs=[mode_radio],
        outputs=[chatbot, history_state, msg_box, mode_radio]
    )

# ==================== 启动应用 ====================
if __name__ == "__main__":
    demo.queue().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="teal")
    )
