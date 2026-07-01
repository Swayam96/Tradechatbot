/**
 * TradeScope-AI chat frontend
 */

(function () {
    "use strict";

    const chatForm = document.getElementById("chat-form");
    const messageInput = document.getElementById("message-input");
    const chatMessages = document.getElementById("chat-messages");
    const typingIndicator = document.getElementById("typing-indicator");
    const sendBtn = document.getElementById("send-btn");
    const sourcesList = document.getElementById("sources-list");

    if (!chatForm) return;

    const BOT_NAME = "TradeScope-AI";

    function escapeHtml(text) {
        const div = document.createElement("div");
        div.textContent = text;
        return div.innerHTML;
    }

    function formatAnswer(text) {
        return escapeHtml(text)
            .replace(/\n\n/g, "</p><p>")
            .replace(/\n/g, "<br>");
    }

    function getTimeString() {
        return new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
    }

    function scrollToBottom() {
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    function appendMessage(content, role, sources) {
        const isUser = role === "user";
        const messageEl = document.createElement("div");
        messageEl.className = `message ${isUser ? "user-message" : "bot-message"}`;

        const avatar = document.createElement("div");
        avatar.className = "message-avatar";
        avatar.innerHTML = isUser
            ? '<i class="bi bi-person-fill"></i>'
            : '<i class="bi bi-stars"></i>';

        const contentWrap = document.createElement("div");
        contentWrap.className = "message-content";

        const meta = document.createElement("div");
        meta.className = "message-meta";
        meta.innerHTML = `
            <span class="message-sender">${isUser ? "You" : BOT_NAME}</span>
            <span class="message-time">${getTimeString()}</span>
        `;

        const bubble = document.createElement("div");
        bubble.className = "message-bubble";

        if (isUser) {
            bubble.innerHTML = `<p class="mb-0">${escapeHtml(content)}</p>`;
        } else {
            bubble.innerHTML = `<p class="mb-0">${formatAnswer(content)}</p>`;

            if (sources && sources.length > 0) {
                const sourcesBlock = document.createElement("div");
                sourcesBlock.className = "message-sources";
                sourcesBlock.innerHTML =
                    '<div class="source-label">Sources</div>' +
                    sources
                        .map(
                            (s) =>
                                `<a href="${escapeHtml(s.url)}" target="_blank" rel="noopener">${escapeHtml(s.title || s.url)}</a>`
                        )
                        .join("");
                bubble.appendChild(sourcesBlock);
            }
        }

        contentWrap.appendChild(meta);
        contentWrap.appendChild(bubble);
        messageEl.appendChild(avatar);
        messageEl.appendChild(contentWrap);
        chatMessages.appendChild(messageEl);
        scrollToBottom();
    }

    function updateSourcesSidebar(sources) {
        if (!sourcesList) return;

        sourcesList.innerHTML = "";

        if (!sources || sources.length === 0) {
            sourcesList.innerHTML = `
                <li class="sources-empty">
                    <i class="bi bi-link-45deg"></i>
                    <span>No sources for this response</span>
                </li>`;
            return;
        }

        sources.forEach((source) => {
            const li = document.createElement("li");
            const title = escapeHtml(source.title || "Source");
            const url = escapeHtml(source.url || "#");
            li.innerHTML = `
                <a href="${url}" target="_blank" rel="noopener">
                    <span class="source-title">${title}</span>
                    <span class="source-url">${url}</span>
                </a>
            `;
            sourcesList.appendChild(li);
        });
    }

    function setLoading(isLoading) {
        if (typingIndicator) {
            typingIndicator.classList.toggle("d-none", !isLoading);
        }
        sendBtn.disabled = isLoading;
        messageInput.disabled = isLoading;
        if (isLoading) scrollToBottom();
    }

    async function sendMessage(message) {
        if (!message.trim()) return;

        appendMessage(message, "user");
        messageInput.value = "";
        setLoading(true);

        try {
            const response = await fetch("/api/chat", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ message }),
            });

            const data = await response.json();

            if (!response.ok && !data.answer) {
                throw new Error(data.error || "Request failed");
            }

            appendMessage(data.answer || "No response received.", "bot", data.sources || []);
            updateSourcesSidebar(data.sources || []);
        } catch (err) {
            appendMessage(
                "Sorry, I couldn't reach the server. Please check your connection and try again.",
                "bot"
            );
            console.error("Chat error:", err);
        } finally {
            setLoading(false);
            messageInput.focus();
        }
    }

    chatForm.addEventListener("submit", (e) => {
        e.preventDefault();
        sendMessage(messageInput.value);
    });

    document.querySelectorAll(".suggestion-chip").forEach((btn) => {
        btn.addEventListener("click", () => {
            const text = btn.textContent.trim();
            sendMessage(text);
        });
    });

    messageInput.focus();
})();
