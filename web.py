import gradio as gr
from main import graph, ChatState


def respond(message, history, mode):
    """
    message: 用户当前输入
    history: 对话历史（列表 of dict，含 role 和 content）
    mode: 当前模式（"导诊建议" 或 "心理咨询"）
    """
    # 1. 转换 history 为 LangGraph 格式
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
        "mode": "1" if mode == "心理咨询" else "2"
    }
    
    # 3. 调用 LangGraph
    try:
        final_state = graph.invoke(state, {"recursion_limit": 50})
        output = final_state.get("final_output")
        current_reply = final_state.get("current_reply")
        
        # 4. 格式化输出
        reply_text = ""
        if output and isinstance(output, dict):
            if "output_body" in output:  # 心理
                reply_text = (
                    f"{output['output_body']}\n\n"
                    f"{output.get('output_advice', '')}\n\n"
                    f"{output.get('soothe_words', '')}"
                )
            elif "text" in output:  # 导诊
                reply_text = output["text"]
            elif "alert_title" in output:  # 急诊
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
        
        # 5. 更新历史（添加用户消息和助手回复）
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": reply_text})
        
        return history, ""  # 返回更新后的历史和空输入框
        
    except Exception as e:
        import traceback
        print("❌ 错误详情:", traceback.format_exc())
        error_msg = f"❌ 发生错误: {e}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, ""


# ---------- 清空历史的回调 ----------
def clear_history():
    return [], ""  # 空历史和空输入框


# ---------- 构建界面 ----------
with gr.Blocks() as demo:
    gr.Markdown("""
    # 🏥 AI 健康导诊助手
    ### 选择模式后开始对话，切换模式会自动清空历史
    """)
    
    # 模式选择
    with gr.Row():
        mode_radio = gr.Radio(
            choices=["导诊建议", "心理咨询"],
            value="导诊建议",
            label="服务模式",
            interactive=True
        )
        clear_btn = gr.Button("🗑️ 清空对话", variant="secondary")
    
    # 状态：存储对话历史
    history_state = gr.State([])
    
    # 聊天显示区域（去掉不支持的参数）
    chatbot = gr.Chatbot(
        label="对话",
        height=400
    )
    
    # 输入区域
    with gr.Row():
        msg_box = gr.Textbox(
            placeholder="请输入您的症状或情绪困扰...",
            scale=4,
            show_label=False,
            lines=2
        )
        send_btn = gr.Button("发送", variant="primary", scale=1)
    
    # ---------- 事件绑定 ----------
    # 1. 发送消息
    def send_message(message, history, mode):
        if not message.strip():
            return history, ""
        return respond(message, history, mode)
    
    send_btn.click(
        send_message,
        inputs=[msg_box, history_state, mode_radio],
        outputs=[chatbot, msg_box]
    )
    
    # 回车发送
    msg_box.submit(
        send_message,
        inputs=[msg_box, history_state, mode_radio],
        outputs=[chatbot, msg_box]
    )
    
    # 2. 清空对话
    clear_btn.click(
        clear_history,
        inputs=[],
        outputs=[chatbot, msg_box]  # 清空显示和输入框
    ).then(
        lambda: ([],),  # 同时重置 history_state
        outputs=[history_state]
    )
    
    # 3. 模式切换 → 清空历史
    def switch_mode(new_mode):
        # 返回空显示、空输入、新模式、空状态
        return [], "", new_mode, []
    
    mode_radio.change(
        switch_mode,
        inputs=[mode_radio],
        outputs=[chatbot, msg_box, mode_radio, history_state]
    )

if __name__ == "__main__":
    demo.queue().launch(
        share=True,
        theme=gr.themes.Soft(primary_hue="blue")  # 主题移到 launch()
    )