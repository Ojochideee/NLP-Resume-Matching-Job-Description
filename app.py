{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "toc_visible": true,
      "authorship_tag": "ABX9TyNdcIe7zXDf5nDjm4EHs1R1",
      "include_colab_link": true
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/Ojochideee/NLP-Resume-Matching-Job-Description/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import warnings\n",
        "warnings.filterwarnings('ignore')"
      ],
      "metadata": {
        "id": "ARemNp-uT8Xv"
      },
      "execution_count": 1,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "import re #import module to work regular expressions\n",
        "import nltk #Python program that can be used for NLP\n",
        "from nltk.corpus import stopwords #WORDS THAT ARE DEEMED INSIGNIFICANT IN TEXT ANALYSIS\n",
        "from nltk.stem import WordNetLemmatizer # reduces words to their root form\n",
        "from nltk.tokenize import word_tokenize\n",
        "import gradio as gr\n",
        "import string\n",
        "from collections import Counter"
      ],
      "metadata": {
        "id": "9udAXGUUaJC3"
      },
      "execution_count": 2,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "# Download NLTK resources (only once)\n",
        "nltk.download('stopwords')\n",
        "nltk.download('punkt')\n",
        "nltk.download('wordnet')\n",
        "nltk.download('punkt_tab')\n",
        "nltk.download('omw-1.4')\n",
        "\n",
        "stop_words = set(stopwords.words('english'))\n",
        "lemmatizer = WordNetLemmatizer()"
      ],
      "metadata": {
        "id": "OgOyTKtHbqgE",
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "outputId": "71305765-13a5-49d3-a97c-4101b2c336fc"
      },
      "execution_count": 3,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stderr",
          "text": [
            "[nltk_data] Downloading package stopwords to /root/nltk_data...\n",
            "[nltk_data]   Package stopwords is already up-to-date!\n",
            "[nltk_data] Downloading package punkt to /root/nltk_data...\n",
            "[nltk_data]   Package punkt is already up-to-date!\n",
            "[nltk_data] Downloading package wordnet to /root/nltk_data...\n",
            "[nltk_data]   Package wordnet is already up-to-date!\n",
            "[nltk_data] Downloading package punkt_tab to /root/nltk_data...\n",
            "[nltk_data]   Package punkt_tab is already up-to-date!\n",
            "[nltk_data] Downloading package omw-1.4 to /root/nltk_data...\n",
            "[nltk_data]   Package omw-1.4 is already up-to-date!\n"
          ]
        }
      ]
    },
    {
      "cell_type": "code",
      "source": [
        "def clean_text(text):\n",
        "    # Lowercase and tokenize\n",
        "    tokens = word_tokenize(text.lower())\n",
        "    # Remove punctuation and stopwords, lemmatize\n",
        "    cleaned = [\n",
        "        lemmatizer.lemmatize(token)\n",
        "        for token in tokens\n",
        "        if token.isalnum() and token not in stop_words\n",
        "    ]\n",
        "\n",
        "    return cleaned"
      ],
      "metadata": {
        "id": "rLws22YBbyOG"
      },
      "execution_count": 4,
      "outputs": []
    },
    {
      "cell_type": "markdown",
      "source": [
        "# GRADIO APP INTERFACE\n"
      ],
      "metadata": {
        "id": "cbdwfdoU3Pbr"
      }
    },
    {
      "cell_type": "code",
      "source": [
        "def compare_resume_to_job(resume_text, job_description_text):\n",
        "    resume_tokens = set(clean_text(resume_text))\n",
        "    job_tokens = clean_text(job_description_text)\n",
        "\n",
        "    # Get top N keywords from job description\n",
        "    job_keywords = [word for word, _ in Counter(job_tokens).most_common(50)]\n",
        "\n",
        "    present = [word for word in job_keywords if word in resume_tokens]\n",
        "    missing = [word for word in job_keywords if word not in resume_tokens]\n",
        "\n",
        "    match_score = round(len(present) / len(job_keywords) * 100, 2)\n",
        "\n",
        "    result = f\"\"\"🔍 **Job Keyword Match Report**\n",
        "\n",
        "✅ Present in Resume ({len(present)}):\n",
        "{', '.join(present)}\n",
        "\n",
        "❌ Missing from Resume ({len(missing)}):\n",
        "{', '.join(missing)}\n",
        "\n",
        "📊 Match Score: **{match_score}%**\n",
        "\"\"\"\n",
        "    return result"
      ],
      "metadata": {
        "id": "px0XzlFGkhxr"
      },
      "execution_count": 7,
      "outputs": []
    },
    {
      "cell_type": "code",
      "source": [
        "gr.Interface(\n",
        "    fn=compare_resume_to_job,\n",
        "    inputs=[\n",
        "        gr.Textbox(lines=15, label=\"Paste your Resume\"),\n",
        "        gr.Textbox(lines=15, label=\"Paste Job Description\"),\n",
        "    ],\n",
        "    outputs=\"text\",\n",
        "    title=\"Resume vs Job Description Keyword Matcher\",\n",
        "    description=\"Paste your resume and job description. It will show you which keywords are missing from your resume.\",\n",
        ").launch(share=True)"
      ],
      "metadata": {
        "id": "0Wy4EKYclUNc",
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 610
        },
        "outputId": "59769cde-3bcf-496c-c7a7-7d230653ffeb"
      },
      "execution_count": 8,
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Colab notebook detected. To show errors in colab notebook, set debug=True in launch()\n",
            "* Running on public URL: https://2ccb7331cbed082462.gradio.live\n",
            "\n",
            "This share link expires in 1 week. For free permanent hosting and GPU upgrades, run `gradio deploy` from the terminal in the working directory to deploy to Hugging Face Spaces (https://huggingface.co/spaces)\n"
          ]
        },
        {
          "output_type": "display_data",
          "data": {
            "text/plain": [
              "<IPython.core.display.HTML object>"
            ],
            "text/html": [
              "<div><iframe src=\"https://2ccb7331cbed082462.gradio.live\" width=\"100%\" height=\"500\" allow=\"autoplay; camera; microphone; clipboard-read; clipboard-write;\" frameborder=\"0\" allowfullscreen></iframe></div>"
            ]
          },
          "metadata": {}
        },
        {
          "output_type": "execute_result",
          "data": {
            "text/plain": []
          },
          "metadata": {},
          "execution_count": 8
        }
      ]
    }
  ]
}