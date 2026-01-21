export class Streamlit {
  static API_VERSION = 1;
  static RENDER_EVENT = "streamlit:render";
  static events = new EventTarget();
  static registeredMessageListener = false;
  static lastFrameHeight = 0;

  static setComponentReady() {
    if (!Streamlit.registeredMessageListener) {
      window.addEventListener("message", Streamlit.onMessageEvent);
      Streamlit.registeredMessageListener = true;
    }
    Streamlit.sendBackMsg("streamlit:componentReady", {
      apiVersion: Streamlit.API_VERSION,
    });
  }

  static setFrameHeight(height) {
    let nextHeight = height;
    if (nextHeight === undefined) {
      nextHeight = document.body.scrollHeight;
    }
    if (nextHeight === Streamlit.lastFrameHeight) {
      return;
    }
    Streamlit.lastFrameHeight = nextHeight;
    Streamlit.sendBackMsg("streamlit:setFrameHeight", { height: nextHeight });
  }

  static setComponentValue(value) {
    Streamlit.sendBackMsg("streamlit:setComponentValue", {
      value: value,
      dataType: "json",
    });
  }

  static onMessageEvent(event) {
    const type = event.data?.type;
    if (type === Streamlit.RENDER_EVENT) {
      Streamlit.onRenderMessage(event.data);
    }
  }

  static onRenderMessage(data) {
    const args = data?.args ?? {};
    const disabled = Boolean(data?.disabled);
    const theme = data?.theme;
    if (theme) {
      injectTheme(theme);
    }
    const event = new CustomEvent(Streamlit.RENDER_EVENT, {
      detail: { disabled, args, theme },
    });
    Streamlit.events.dispatchEvent(event);
  }

  static sendBackMsg(type, data) {
    window.parent.postMessage({ isStreamlitMessage: true, type, ...data }, "*");
  }
}

function injectTheme(theme) {
  const style = document.createElement("style");
  document.head.appendChild(style);
  style.innerHTML = `
    :root {
      --primary-color: ${theme.primaryColor};
      --background-color: ${theme.backgroundColor};
      --secondary-background-color: ${theme.secondaryBackgroundColor};
      --text-color: ${theme.textColor};
      --font: ${theme.font};
    }

    body {
      background-color: var(--background-color);
      color: var(--text-color);
    }
  `;
}
