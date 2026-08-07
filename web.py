import gradio as gr
from main import graph, ChatState
import numpy as np
import soundfile as sf
import tempfile
import os

# ==================== 语音识别模块（如果不需要可删除） ====================
# 若未安装 faster-whisper 或不想用语音，可注释掉以下导入和模型加载
try:
    ##from faster_whisper import WhisperModel
    ##whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    VOICE_ENABLED = False
except ImportError:
    VOICE_ENABLED = False
    print("⚠️ faster-whisper 未安装，语音功能不可用")

def transcribe_audio(audio):
    if not VOICE_ENABLED or audio is None:
        return ""
    sr_rate, y = audio
    if y.ndim > 1:
        y = y.mean(axis=1)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
        sf.write(tmp_path, y, sr_rate)
    try:
        segments, _ = whisper_model.transcribe(tmp_path, language="zh", beam_size=3)
        text = " ".join([seg.text for seg in segments])
    except Exception as e:
        print("转写错误:", e)
        text = ""
    finally:
        os.unlink(tmp_path)
    return text

# ==================== 核心回复处理（与 main.py 交互） ====================
def respond(message, history, mode):
    # 将 Gradio 的 mode（中文或英文标签）转换为内部 mode 标识
    if "心理" in mode or "Mental" in mode:
        mode_flag = "1"  # 心理
    else:
        mode_flag = "2"  # 导诊

    chat_history = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if content:
            chat_history.append({"role": role, "content": content})

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
        "mode": mode_flag
    }

    try:
        final_state = graph.invoke(state, {"recursion_limit": 50})
        output = final_state.get("final_output")
        current_reply = final_state.get("current_reply")

        reply_text = ""
        if output and isinstance(output, dict):
            if "output_body" in output:
                reply_text = (
                    f"{output['output_body']}\n\n"
                    f"{output.get('output_advice', '')}\n\n"
                    f"{output.get('soothe_words', '')}"
                )
            elif "text" in output:
                reply_text = output["text"]
            elif "alert_title" in output:
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
            reply_text = "系统暂时无法处理，请稍后再试。 / System temporarily unavailable. Please try again."

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
def send_message(message, history, mode):
    if not message.strip():
        return history, history, ""
    new_history, _ = respond(message, history, mode)
    return new_history, new_history, ""

def clear_history():
    return [], [], ""

def switch_mode(new_mode):
    # 切换模式时清空历史
    return [], [], "", new_mode

def process_voice(audio, history, mode):
    if not VOICE_ENABLED:
        return history, history, "", None
    text = transcribe_audio(audio)
    if not text.strip():
        return history, history, "", None
    # 语音识别后自动填入输入框，不自动发送（用户可编辑后手动发送）
    return history, history, text, None

# ==================== 构建 Gradio 界面 ====================

# 自定义 CSS（可选），保持柔和风格
custom_css = """
    .gradio-container {
        background: linear-gradient(145deg, #fdf6f0 0%, #f0e6de 100%);
        font-family: 'Georgia', serif;
    }
    .message {
        border-radius: 20px !important;
        padding: 15px 20px !important;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05) !important;
    }
    .user .message {
        background-color: #d4c5b2 !important;
        color: #3d2c1e !important;
    }
    .bot .message {
        background-color: #ffffff !important;
        color: #4a3728 !important;
        border: 1px solid #e8ddd0 !important;
    }
    textarea, .input-box {
        border-radius: 30px !important;
        border: 1px solid #dccfc3 !important;
        background-color: #ffffff !important;
    }
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

with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="stone")) as demo:
    # ========== 顶部：语言切换 + 提示 ==========
    with gr.Row():
        lang_radio = gr.Radio(
            choices=["中文", "English"],
            value="中文",
            label="🌐 界面语言 / Interface Language",
            interactive=True
        )
        # 动态提示信息
        hint_md = gr.Markdown(
            "💬 AI 将根据您的输入语言自动回复（中/英） / AI will reply in the language you use (Chinese/English)"
        )

    # 更新提示信息
    def update_hint(lang):
        if lang == "中文":
            return "💬 AI 将根据您的输入语言自动回复（中/英）"
        else:
            return "💬 AI will reply in the language you use (Chinese/English)"
    lang_radio.change(update_hint, inputs=lang_radio, outputs=hint_md)

    # ========== 标题 ==========
    gr.Markdown("""
    # 🏥 AI 健康导诊助手 · AI Health Triage Assistant
    ### 选择模式后开始对话，切换模式会自动清空历史
    ### Select a mode to start. Switching modes will clear history.
    """)

    # ========== 模式选择和清空按钮 ==========
    with gr.Row():
        mode_radio = gr.Radio(
            choices=["导诊建议 / Triage", "心理咨询 / Mental Health"],
            value="导诊建议 / Triage",
            label="服务模式 / Service Mode",
            interactive=True
        )
        clear_btn = gr.Button("🗑️ 清空对话 / Clear Chat", variant="secondary")

    # ========== 聊天显示 ==========
    chatbot = gr.Chatbot(label="对话 / Chat", height=400)
    history_state = gr.State([])

    # ========== 输入区域 ==========
    with gr.Row():
        # 如果有语音功能则显示音频输入，否则隐藏
        if VOICE_ENABLED:
            audio_input = gr.Audio(
                sources=["microphone"],
                type="numpy",
                label="🎤 按住说话 / Hold to speak",
                show_label=True,
                scale=1
            )
        else:
            audio_input = None

        msg_box = gr.Textbox(
            placeholder="请输入您的症状或情绪困扰... / Enter your symptoms or concerns...",
            scale=4,
            show_label=False,
            lines=2
        )
        send_btn = gr.Button("发送 / Send", variant="primary", scale=1)

    # ========== 事件绑定 ==========
    # 发送文本
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

    # 语音输入（如果启用）
    if VOICE_ENABLED and audio_input is not None:
        audio_input.stop_recording(
            process_voice,
            inputs=[audio_input, history_state, mode_radio],
            outputs=[chatbot, history_state, msg_box, audio_input]
        )

    # 清空对话
    clear_btn.click(
        clear_history,
        outputs=[chatbot, history_state, msg_box]
    )

    # 模式切换 -> 清空历史
    mode_radio.change(
        switch_mode,
        inputs=[mode_radio],
        outputs=[chatbot, history_state, msg_box, mode_radio]
    )

# ==================== 启动 ====================
if __name__ == "__main__":
    demo.queue().launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="stone")
    )
