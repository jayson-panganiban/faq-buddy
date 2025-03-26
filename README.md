# 🚗 FAQ Bot - Your Car Insurance Chatbot

Welcome to the FAQ Bot! This is a simple chatbot designed to help you with questions about car insurance in Australia. It uses a collection of Frequently Asked Questions (FAQs) and OpenAI's technology to provide you with helpful answers.

## ✨ Features

- **Smart Matching**: The bot understands your questions and finds the best answers from its FAQ database.
- **Dynamic Responses**: If it can't find an exact match, it will create a helpful answer based on similar questions.
- **User-Friendly**: Just type your question, and the bot will respond instantly!

## 📋 What You Need to Get Started

Before you begin, make sure you have:

- **Git**: Version control software to manage your code. You can download it from [Git's official website](https://git-scm.com/downloads).
- **Python**: Version 3.7 or higher installed on your computer. You can download it from [Python's official website](https://www.python.org/downloads/).
- **OpenAI API Key**: You'll need an API key from OpenAI to use the chatbot. You can sign up for one [here](https://platform.openai.com/signup).

## 🚀 Step-by-Step Setup Guide

### 1️⃣ Clone the Repository

Open your terminal (Command Prompt on Windows) and run the following commands:

```bash
git clone https://github.com/Andreymae/faq-bot.git
cd faq-bot
```

If you don't have Git installed, you can download the project as a ZIP file from [this link](https://github.com/Andreymae/faq-bot/archive/refs/heads/main.zip) and extract it.

### 2️⃣ Create a Virtual Environment

Setting up a virtual environment is a good practice to keep your project organized. It creates a separate space for your project dependencies.

**For Mac/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**For Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

You should see `(venv)` at the beginning of your command line, indicating that the virtual environment is active.

### 3️⃣ Install Required Packages

With your virtual environment activated, install the necessary packages by running:

```bash
pip install -r requirements.txt
```

This command will download all the libraries bot needs to function.

### 4️⃣ Set Up Your OpenAI API Key

The bot needs an OpenAI API key to work properly. Set it as an environment variable.

**For Mac/Linux:**
```bash
export OPENAI_API_KEY=your_openai_api_key_here
```

**For Windows:**
```bash
set OPENAI_API_KEY=your_openai_api_key_here
```

Make sure to replace `your_openai_api_key_here` with your actual API key from OpenAI.

### 5️⃣ Run the Bot

Now, let's start the bot! Run the following command in your terminal:

```bash
python main.py
```

You should see messages indicating that the server is running.

### 6️⃣ Interact with the Bot

1. Open your web browser and go to **http://localhost:8000**.
2. Enter your OpenAI API key in the settings panel.
3. Start asking questions about car insurance in Australia!

## 💬 Special Commands

You can use these special commands while chatting with the bot:

- `/reload` - Refresh the FAQ database.
- `/sources` - See the sources of the information.
- `/health` - Check if the bot is functioning properly.
- `/about` - Learn more about the bot and its creators.

## 📚 Additional Resources

- Explore the API documentation at **http://localhost:8000/docs**.
- Check out the source code to see how everything works.
- Try asking different questions to see how the bot responds!

## ❓ Troubleshooting Tips

- **Bot won't start?** Ensure Python is installed correctly and that you're in the right directory.
- **Not getting answers?** Double-check that your OpenAI API key is entered correctly.
- **Encountering errors?** Make sure all required packages were installed successfully.


---

Happy chatting with your new car insurance assistant! 🚗💬
