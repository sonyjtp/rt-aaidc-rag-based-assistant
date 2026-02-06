# UI Guide: RAG-Based Chatbot Web Interface

## Overview

The RAG-Based Assistant includes a modern, intuitive web interface built with **Streamlit**. This guide explains all features, components, and how to interact with the application.

## Launching the UI

### Start the Streamlit App

```bash
streamlit run src/streamlit_app.py
```

The application will launch at `http://localhost:8501` by default.

---

## Main Components

### 1. **Page Configuration**

The application automatically configures itself on startup:

- **Page Title**: "🤖 RAG-Based Chatbot"
- **Custom Styling**: Dark theme with optimized colors for readability
- **Responsive Layout**: Adapts to different screen sizes

### 2. **Sidebar Controls**

The left sidebar (`⚙️ Configuration`) provides essential controls:

#### **Clear Chat History Button**
- **Icon**: 🗑️
- **Function**: Clears all conversation history instantly
- **Use Case**: Start a fresh conversation or reset the chat state
- **Status**: Full-width button for easy access

#### **Status Section**
- **Location**: Below the Clear button
- **Display States**:
  - ✅ **Green "RAG Assistant Ready"**: System is initialized and ready for questions
  - ⏳ **Yellow "Initializing assistant..."**: System is still loading documents and models

---

## Chat Interface

### 3. **Main Content Area**

#### **Header Section**
```
🤖 RAG-Based Chatbot
This chatbot uses Retrieval-Augmented Generation (RAG) to answer questions
based on a collection of documents. Ask any question about the documents below!
```

Explains the core functionality to new users.

### 4. **Chat History Display**

Only appears after the first message is sent.

#### **Message Rendering**
- **User Messages**: Styled with a distinct background color, prefixed with "You:"
- **Assistant Messages**: Different styling, prefixed with "Assistant:"
- **Formatted Display**: Uses HTML styling for visual distinction
- **Scrollable Container**: Chat history scrolls if it exceeds viewport height

#### **Example Chat Display**
```
┌─────────────────────────────────────┐
│ You:                                │
│ What are the major religions of India? │
├─────────────────────────────────────┤
│ Assistant:                          │
│ India is home to multiple major     │
│ religions including Hinduism,       │
│ Buddhism, Sikhism, and more...      │
└─────────────────────────────────────┘
```

### 5. **Input Area**

#### **Question Input Field**
- **Label**: "Enter your question about the documents:"
- **Placeholder**: "e.g., What are the major rivers in India? Or ask about Indian history, culture, etc."
- **Help Text** (on hover): "Ask any question about India—history, culture, geography, religions, government, economy, cuisine, sports, and more."
- **Input Type**: Text input with automatic focus
- **Submission**:
  - Press `Enter` or click the "Send" button
  - Supports multi-line input (use `Shift+Enter`)

#### **Send Button**
- **Icon**: None (text-based)
- **Style**: Primary blue button
- **Position**: Right-aligned next to input field
- **Behavior**:
  - Disabled until user enters text
  - Clears input after submission

---

## Processing Workflow

### 6. **Response Generation Status**

When you send a question, the app displays a status indicator:

```
🔍 Searching documents and generating response...
```

#### **Status States**
- **Processing** (🔍): Documents are being searched and response is being generated
- **Complete** (✅): Response successfully generated
- **Error** (❌): An error occurred; user is prompted to try again

#### **Behind the Scenes**
1. Document retrieval from vector database
2. RAG-enhanced reasoning applied
3. Response validation and filtering
4. Markdown cleanup (removes headers, separators)
5. Topic validation (only knowledge-base topics shown)

---

## Features & Capabilities

### 7. **Automatic Initialization**

The application automatically:
- Loads all documents from the `data/` directory
- Initializes the RAG Assistant
- Creates vector embeddings for semantic search
- Displays a status message while initializing

**What happens if initialization fails?**
- Error notification appears
- User is prompted to refresh and try again
- Contact support message is shown

### 8. **Chat History Management**

- **Persistent in Session**: History stays during your session
- **Cleared on Button Click**: Can be reset anytime via sidebar button
- **Auto-Rerun**: UI refreshes after each exchange
- **No Persistence**: History is lost when you close the browser

### 9. **Response Processing**

The assistant performs several steps on each response:

1. **Topic Validation**: Filters to only include knowledge-base topics
2. **Markdown Cleanup**:
   - Removes markdown headers (`#`, `##`, etc.)
   - Removes bold markers (`**`)
   - Removes separator lines
   - Cleans up extra whitespace
3. **Logging**: All responses are logged for debugging

---

## User Interactions & Workflows

### Workflow 1: **Basic Question**

```
1. Type: "What is AI?"
2. Click "Send" or press Enter
3. App searches documents (status shows 🔍)
4. Response appears with ✅ status
5. Message added to chat history
```

### Workflow 2: **Multi-turn Conversation**

```
1. Ask: "What is the history of India?"
2. Follow-up: "Tell me more about the freedom struggle"
3. Each turn adds to chat history
4. Click "Clear Chat History" to reset
5. Sidebar shows ✅ Ready status throughout
```

### Workflow 3: **Error Recovery**

```
1. If response fails (❌ status)
2. Error message is displayed
3. User can try again with a different question
4. No need to refresh (error is localized)
```

---

## Styling & Customization

### 10. **Visual Design**

#### **CSS Classes**
- `.title-text`: Large, bold title
- `.chat-message`: Container for each message
- `.user-message`: Styled for user inputs (light background)
- `.assistant-message`: Styled for assistant responses (different color)

#### **Color Scheme**
- **Primary**: Streamlit blue for buttons and accents
- **Success**: Green for "Ready" status
- **Warning**: Yellow for "Initializing" status
- **Error**: Red for error messages

#### **Responsive Layout**
- **Input Area**: Uses 5:1 column ratio (input:button)
- **Full-Width Buttons**: Sidebar buttons span entire width
- **Scrollable Containers**: Chat history scrolls vertically
- **Mobile-Friendly**: Adapts to smaller screens

---

## Configuration Files

### 11. **Related Configuration**

The UI behavior is influenced by:

1. **`config/prompt-config.yaml`**: System prompts and instructions
2. **`config/reasoning_strategies.yaml`**: Response reasoning approach
3. **`config/memory_strategies.yaml`**: Conversation memory management
4. **`data/` directory**: Knowledge base documents

See main [README.md](README.md) for detailed configuration documentation.

---

## Troubleshooting

### Issue: "⏳ Initializing assistant..." shows indefinitely

**Solution**:
- Refresh the browser page
- Check that documents exist in `data/` folder
- Verify model files are downloaded

### Issue: Error message appears after sending question

**Solution**:
- Click "Clear Chat History"
- Try a simpler question
- Check logs for detailed error messages

### Issue: "Unable to initialize the assistant at this time"

**Solution**:
- Refresh the page
- Check internet connection
- Verify `.env` file has correct API keys
- See [README.md](README.md) for setup instructions

---

## Performance Tips

1. **Keep Chat History Clean**: Use "Clear Chat History" to remove old exchanges
2. **Clear Questions**: Ask specific, well-formed questions
3. **Document Size**: Smaller, well-organized documents are processed faster
4. **GPU Usage**: Ensure GPU is available for faster embeddings (see README)

---

## Next Steps

- **Customize Responses**: Edit `config/reasoning_strategies.yaml`
- **Add Documents**: Place `.txt` files in the `data/` folder
- **Adjust Prompts**: Modify `config/prompt-config.yaml`
- **Memory Management**: Configure `config/memory_strategies.yaml`

For more details, see the main [README.md](README.md).
