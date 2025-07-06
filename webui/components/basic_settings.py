import traceback

import streamlit as st
import os
from app.config import config
from app.utils import utils
from loguru import logger


def render_basic_settings(tr):
    """渲染基础设置面板"""
    with st.expander(tr("Basic Settings"), expanded=False):
        config_panels = st.columns(3)
        left_config_panel = config_panels[0]
        middle_config_panel = config_panels[1]
        right_config_panel = config_panels[2]

        with left_config_panel:
            render_language_settings(tr)
            render_proxy_settings(tr)

        with middle_config_panel:
            render_vision_llm_settings(tr)  # 视频分析模型设置

        with right_config_panel:
            render_text_llm_settings(tr)  # 文案生成模型设置


def render_language_settings(tr):
    st.subheader(tr("Proxy Settings"))

    """渲染语言设置"""
    system_locale = utils.get_system_locale()
    i18n_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "i18n")
    locales = utils.load_locales(i18n_dir)

    display_languages = []
    selected_index = 0
    for i, code in enumerate(locales.keys()):
        display_languages.append(f"{code} - {locales[code].get('Language')}")
        if code == st.session_state.get('ui_language', system_locale):
            selected_index = i

    selected_language = st.selectbox(
        tr("Language"),
        options=display_languages,
        index=selected_index
    )

    if selected_language:
        code = selected_language.split(" - ")[0].strip()
        st.session_state['ui_language'] = code
        config.ui['language'] = code


def render_proxy_settings(tr):
    """渲染代理设置"""
    # 获取当前代理状态
    proxy_enabled = config.proxy.get("enabled", False)
    proxy_url_http = config.proxy.get("http")
    proxy_url_https = config.proxy.get("https")

    # 添加代理开关
    proxy_enabled = st.checkbox(tr("Enable Proxy"), value=proxy_enabled)
    
    # 保存代理开关状态
    # config.proxy["enabled"] = proxy_enabled

    # 只有在代理启用时才显示代理设置输入框
    if proxy_enabled:
        HTTP_PROXY = st.text_input(tr("HTTP_PROXY"), value=proxy_url_http)
        HTTPS_PROXY = st.text_input(tr("HTTPs_PROXY"), value=proxy_url_https)

        if HTTP_PROXY and HTTPS_PROXY:
            config.proxy["http"] = HTTP_PROXY
            config.proxy["https"] = HTTPS_PROXY
            os.environ["HTTP_PROXY"] = HTTP_PROXY
            os.environ["HTTPS_PROXY"] = HTTPS_PROXY
            # logger.debug(f"代理已启用: {HTTP_PROXY}")
    else:
        # 当代理被禁用时，清除环境变量和配置
        os.environ.pop("HTTP_PROXY", None)
        os.environ.pop("HTTPS_PROXY", None)
        # config.proxy["http"] = ""
        # config.proxy["https"] = ""


def test_vision_model_connection(api_key, base_url, model_name, provider, tr):
    """测试视觉模型连接
    
    Args:
        api_key: API密钥
        base_url: 基础URL
        model_name: 模型名称
        provider: 提供商名称
    
    Returns:
        bool: 连接是否成功
        str: 测试结果消息
    """
    if provider.lower() == 'gemini':
        import google.generativeai as genai
        
        try:
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(model_name)
            model.generate_content("直接回复我文本'当前网络可用'")
            return True, tr("gemini model is available")
        except Exception as e:
            return False, f"{tr('gemini model is not available')}: {str(e)}"
    elif provider.lower() == 'narratoapi':
        import requests
        try:
            # 构建测试请求
            headers = {
                "Authorization": f"Bearer {api_key}"
            }
        
            test_url = f"{base_url.rstrip('/')}/health"
            response = requests.get(test_url, headers=headers, timeout=10)
        
            if response.status_code == 200:
                return True, tr("NarratoAPI is available")
            else:
                return False, f"{tr('NarratoAPI is not available')}: HTTP {response.status_code}"
        except Exception as e:
            return False, f"{tr('NarratoAPI is not available')}: {str(e)}"

    else:
        from openai import OpenAI
        try:
            client = OpenAI(
                api_key=api_key,
                base_url=base_url,
            )

            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": "You are a helpful assistant."}],
                    },
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://help-static-aliyun-doc.aliyuncs.com/file-manage-files/zh-CN/20241022/emyrja/dog_and_girl.jpeg"
                                },
                            },
                            {"type": "text", "text": "回复我网络可用即可"},
                        ],
                    },
                ],
            )
            if response and response.choices:
                return True, tr("QwenVL model is available")
            else:
                return False, tr("QwenVL model returned invalid response")

        except Exception as e:
            # logger.debug(api_key)
            # logger.debug(base_url)
            # logger.debug(model_name)
            return False, f"{tr('QwenVL model is not available')}: {str(e)}"


def render_vision_llm_settings(tr):
    """渲染视频分析模型设置"""
    st.subheader(tr("Vision Model Settings"))

    # 视频分析模型提供商选择
    vision_providers = ['Siliconflow', 'Gemini', 'QwenVL', 'OpenAI']
    saved_vision_provider = config.app.get("vision_llm_provider", "Gemini").lower()
    saved_provider_index = 0

    for i, provider in enumerate(vision_providers):
        if provider.lower() == saved_vision_provider:
            saved_provider_index = i
            break

    vision_provider = st.selectbox(
        tr("Vision Model Provider"),
        options=vision_providers,
        index=saved_provider_index
    )
    vision_provider = vision_provider.lower()
    config.app["vision_llm_provider"] = vision_provider
    st.session_state['vision_llm_providers'] = vision_provider

    # 获取已保存的视觉模型配置
    vision_api_key = config.app.get(f"vision_{vision_provider}_api_key", "")
    vision_base_url = config.app.get(f"vision_{vision_provider}_base_url", "")
    vision_model_name = config.app.get(f"vision_{vision_provider}_model_name", "")

    # 渲染视觉模型配置输入框
    st_vision_api_key = st.text_input(tr("Vision API Key"), value=vision_api_key, type="password")
    
    # 根据不同提供商设置默认值和帮助信息
    if vision_provider == 'gemini':
        st_vision_base_url = st.text_input(
            tr("Vision Base URL"), 
            value=vision_base_url,
            disabled=True,
            help=tr("Gemini API does not require a base URL")
        )
        st_vision_model_name = st.text_input(
            tr("Vision Model Name"), 
            value=vision_model_name or "gemini-2.0-flash-lite",
            help=tr("Default: gemini-2.0-flash-lite")
        )
    elif vision_provider == 'qwenvl':
        st_vision_base_url = st.text_input(
            tr("Vision Base URL"), 
            value=vision_base_url,
            help=tr("Default: https://dashscope.aliyuncs.com/compatible-mode/v1")
        )
        st_vision_model_name = st.text_input(
            tr("Vision Model Name"), 
            value=vision_model_name or "qwen-vl-max-latest",
            help=tr("Default: qwen-vl-max-latest")
        )
    else:
        st_vision_base_url = st.text_input(tr("Vision Base URL"), value=vision_base_url)
        st_vision_model_name = st.text_input(tr("Vision Model Name"), value=vision_model_name)

    # 在配置输入框后添加测试按钮
    if st.button(tr("Test Connection"), key="test_vision_connection"):
        with st.spinner(tr("Testing connection...")):
            success, message = test_vision_model_connection(
                api_key=st_vision_api_key,
                base_url=st_vision_base_url,
                model_name=st_vision_model_name,
                provider=vision_provider,
                tr=tr
            )
            
            if success:
                st.success(tr(message))
            else:
                st.error(tr(message))

    # 保存视觉模型配置
    if st_vision_api_key:
        config.app[f"vision_{vision_provider}_api_key"] = st_vision_api_key
        st.session_state[f"vision_{vision_provider}_api_key"] = st_vision_api_key
    if st_vision_base_url:
        config.app[f"vision_{vision_provider}_base_url"] = st_vision_base_url
        st.session_state[f"vision_{vision_provider}_base_url"] = st_vision_base_url
    if st_vision_model_name:
        config.app[f"vision_{vision_provider}_model_name"] = st_vision_model_name
        st.session_state[f"vision_{vision_provider}_model_name"] = st_vision_model_name


def test_text_model_connection(api_key, base_url, model_name, provider, tr):
    """测试文本模型连接
    
    Args:
        api_key: API密钥
        base_url: 基础URL
        model_name: 模型名称
        provider: 提供商名称
    
    Returns:
        bool: 连接是否成功
        str: 测试结果消息
    """
    import requests
    
    try:
        # 构建统一的测试请求（遵循OpenAI格式）
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

        # 特殊处理Gemini
        if provider.lower() == 'gemini':
            import google.generativeai as genai
            try:
                genai.configure(api_key=api_key)
                model = genai.GenerativeModel(model_name)
                model.generate_content("直接回复我文本'当前网络可用'")
                return True, tr("Gemini model is available")
            except Exception as e:
                return False, f"{tr('Gemini model is not available')}: {str(e)}"
        else:
            test_url = f"{base_url.rstrip('/')}/chat/completions"

            # 构建测试消息
            test_data = {
                "model": model_name,
                "messages": [
                    {"role": "user", "content": "直接回复我文本'当前网络可用'"}
                ],
                "stream": False
            }

            # 发送测试请求
            response = requests.post(
                test_url,
                headers=headers,
                json=test_data,
            )
            # logger.debug(model_name)
            # logger.debug(api_key)
            # logger.debug(test_url)
            if response.status_code == 200:
                return True, tr("Text model is available")
            else:
                return False, f"{tr('Text model is not available')}: HTTP {response.status_code}"
            
    except Exception as e:
        logger.error(traceback.format_exc())
        return False, f"{tr('Connection failed')}: {str(e)}"


def render_text_llm_settings(tr):
    """渲染文案生成模型设置"""
    st.subheader(tr("Text Generation Model Settings"))

    # 文案生成模型提供商选择
    text_providers = ['OpenAI', 'Siliconflow', 'DeepSeek', 'Gemini', 'Qwen', 'Moonshot']
    saved_text_provider = config.app.get("text_llm_provider", "OpenAI").lower()
    saved_provider_index = 0

    for i, provider in enumerate(text_providers):
        if provider.lower() == saved_text_provider:
            saved_provider_index = i
            break

    text_provider = st.selectbox(
        tr("Text Model Provider"),
        options=text_providers,
        index=saved_provider_index
    )
    text_provider = text_provider.lower()
    config.app["text_llm_provider"] = text_provider

    # 获取已保存的文本模型配置
    text_api_key = config.app.get(f"text_{text_provider}_api_key")
    text_base_url = config.app.get(f"text_{text_provider}_base_url")
    text_model_name = config.app.get(f"text_{text_provider}_model_name")

    # 渲染文本模型配置输入框
    st_text_api_key = st.text_input(tr("Text API Key"), value=text_api_key, type="password")
    st_text_base_url = st.text_input(tr("Text Base URL"), value=text_base_url)
    st_text_model_name = st.text_input(tr("Text Model Name"), value=text_model_name)

    # 添加测试按钮
    if st.button(tr("Test Connection"), key="test_text_connection"):
        with st.spinner(tr("Testing connection...")):
            success, message = test_text_model_connection(
                api_key=st_text_api_key,
                base_url=st_text_base_url,
                model_name=st_text_model_name,
                provider=text_provider,
                tr=tr
            )
            
            if success:
                st.success(message)
            else:
                st.error(message)

    # 保存文本模型配置
    if st_text_api_key:
        config.app[f"text_{text_provider}_api_key"] = st_text_api_key
    if st_text_base_url:
        config.app[f"text_{text_provider}_base_url"] = st_text_base_url
    if st_text_model_name:
        config.app[f"text_{text_provider}_model_name"] = st_text_model_name

    # # Cloudflare 特殊配置
    # if text_provider == 'cloudflare':
    #     st_account_id = st.text_input(
    #         tr("Account ID"),
    #         value=config.app.get(f"text_{text_provider}_account_id", "")
    #     )
    #     if st_account_id:
    #         config.app[f"text_{text_provider}_account_id"] = st_account_id
