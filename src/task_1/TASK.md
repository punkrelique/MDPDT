
# Task objective: learn how to create applications using ready-made machine learning libraries.

Task concept (this is an intro, not the text of the task itself):
As part of this task, you need to familiarize yourself with the capabilities of AI/ML/DL for solving applied problems, determine which applied problems can be solved using machine learning libraries and models, choose the 4 most interesting problems + LLM for yourself and solve them using existing trained models.
For example, you can solve the following 4 problems:
– determining the tonality of texts,
– image classification;
– the task of translating audio into text;
– detecting objects in video.
It is worth identifying the most interesting task options for yourself.
Please note that it is necessary to take into account that an API will need to be developed for one of the four models later.
You also need to select an LLM that can be deployed on your infrastructure/locally on your computer and deploy it) Most likely, this will be model 2/3/7/8B.
Task description:
1. Create a repository on GitHub.
2. Study the capabilities of ready-made machine learning libraries.
3. Formulate the problem that you want to solve using machine learning tools and briefly describe the technical task (specification).
4. Implement the solution to the problem you have chosen using a ready-made machine learning library.
5. Upload the implemented solution to the repository on GitHub.
6. Document your solution in the repository.
7. (!) Complete steps 3-6 as part of developing a software solution for the following 4 tasks with different types of content:
1. text processing (e.g. natural language sentiment analysis, text generation, ...)
2. audio processing (speech synthesis, speech-to-text, command detection, ...)
3. image processing (classification, object detection in images, image quality improvement, ...)
4. video processing (object detection, gesture recognition, ...)
+5. LLM

Solutions must be implemented using at least 2 framework options.
For example:
1. natural language processing (Hugging Face)
2. audio processing (TensorFlow)
3. image processing (PyTorch)
4. video processing (Hugging Face)
+5. LLM (open source, no special requirements, just that it works locally (not on third-party providers' servers)

That is, you work only with ready-made solutions. You don't need to train models/retrain (fine tune) LLMs - you just need to understand the capabilities of AI today and deploy them on your computer (locally, not WEB), wrappers/frontends are also not required (but are not prohibited)
_________
Based on the results of the task, a Report is generated (including a description of
1. the advantages of the selected library/solution,
2. the principle of operation of the Model and the reasons for its selection,
3. the structure of the dataset (for training/for inference),
4. metrics for assessing the quality of the model (how to assess that the model is working properly? You can track ML metrics (for example, Accuracy, Recall ...), product metrics (Session time), infrastructure metrics (for example, GPU usage)),
5. features of the implementation of the algorithm, its efficiency).
It is recommended to use several of the following machine learning libraries training:
Hugging Face (https://huggingface.co/ ),
TensorFlow Hub (https://www.tensorflow.org/hub ),
PyTorch Hub (https://pytorch.org/hub/ ),
Keras Applications (https://keras.io/api/applications/ ).
Sample repository – https://github.com/sozykin/ml_sentiment
