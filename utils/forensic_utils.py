from .utils import DEFAULT_IMAGE_TOKEN

LONG_QUESTION_LIST = [
    f"{DEFAULT_IMAGE_TOKEN}\nTask: Analyze the image. Please provide a segmentation mask and determine the image's authenticity.",
    f"{DEFAULT_IMAGE_TOKEN}\nTask: Image analysis. Generate a segmentation mask and verify if the image is authentic.",
    f"Task: Image verification. Is the image {DEFAULT_IMAGE_TOKEN} genuine or altered? If altered, mask the tampered areas."
]


EXPLANATORY_QUESTION_LIST = [
    "Please output a segmentation mask, verify its authenticity",
    "Please output a segmentation mask, confirm its authenticity",
    "Please output a segmentation mask, check its authenticity",
]

ANSWER_LIST = [
    "It is [SEG] and [DET].",
    "Sure, [SEG] and [DET].",
    "Sure, it is [SEG] and [DET].",
    "Sure, the segmentation result is [SEG] and [DET].",
    "[SEG] and [DET].",
]
