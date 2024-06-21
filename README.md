# Image-Caption-Generation-using-Gemini
we generate captions to the images which are given by user(user input) using prompt engineering and Generative AI
# Vision Models
Vision models can look at pictures and then tell you what's in them using words. These are called vision-to-text models. They bring together the power of understanding images and language. Using fancy neural networks, these models can look at pictures and describe them in a way that makes sense. They're like a bridge between what you see and what you can read.

This is super useful for things like making captions for images, helping people who can't see well understand what's in a picture, and organizing information. As these models get even smarter, they're going to make computers even better at understanding and talking about what they "see" in pictures. It's like teaching computers to understand and describe the visual world around us.![image](https://github.com/Pavansomisetty21/Image-Caption-Generation-using-Gemini/assets/110320361/b8b48459-ecfb-42eb-b379-e82543a8334f)
Based on which model we are using like OPENAI,GEMINI(sub models),Anthropic we use respective API KEY and also we use endpoint if necessary 
# Gemini compared to other Foundation Models
Evidence suggests Gemini represents the state-of-the-art in foundation models:

It achieves record-breaking results on over 56 benchmarks spanning text, code, math, visual, and audio understanding. This includes benchmarks like MMLU, GSM8K, MATH, Big-Bench Hard, HumanEval, Natural2Code, DROP, and WMT23.

➡️Notably, Gemini Ultra is the first to achieve human-expert performance on MMLU across 57 subjects with scores above 90%.

➡️On conceptual reasoning benchmarks like BIG-Bench, Gemini outperforms expert humans in areas like math, physics, and CS.

➡️Specialized versions create state-of-the-art applications like the code generator AlphaCode 2 which solves programming problems better than 85% of human coders in competitions.

➡️Qualitative examples show Gemini can manipulate complex math symbols, correct errors in derivations, generate relevant UIs based on conversational context, and more.
# Multi-modal architectures
Google does not disclose details of Gemini architecture. But some multi-modal architectures shared by research community give us some intuitions on how it might work.
The CLIP architecture(https://openai.com/research/clip), introduced in 2021, uses contrastive learning between images with textual representations, with some distance function like cosine similarity to align the embedding spaces.
![image](https://github.com/Pavansomisetty21/Image-Caption-Generation-using-Gemini/assets/110320361/936ee54f-8143-4971-ae23-c75625ee545f)

➡️Flamingo(https://arxiv.org/abs/2204.14198) uses a vision encoder pre-trained using CLIP and a Chinchilla(https://arxiv.org/abs/2203.15556) pre-trained language model to represent the text. It introduces some special components — the Perceiver Resampler and a special Gated cross-attention — to combine those interleaved multi-modal representations, and is trained to predict next tokens. Flamingo can perform visual question answering or conversations around that content.
![image](https://github.com/Pavansomisetty21/Image-Caption-Generation-using-Gemini/assets/110320361/99b19882-8e9f-4ee7-abbf-cdfae495f99c)
The above diagram represents that examples of visual dialogue from Flamingo paper

➡️BLIP-2 (https://arxiv.org/abs/2301.12597) also uses pre-trained image and LLM encoders, connected by a Q-Former component. The model is trained for multiple tasks: matching images and text representations with both constrastive learning (like CLIP) and with binary classification task. It is also trained on images caption generation.
The illustration for this paper will be as ![image](https://github.com/Pavansomisetty21/Image-Caption-Generation-using-Gemini/assets/110320361/73ac1ea8-9778-48e6-8f8d-f0c3516f5afe)

