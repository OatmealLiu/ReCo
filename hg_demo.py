import torch
from diffusers import StableDiffusionPipeline


def create_reco_prompt(
        caption: str = '',
        phrases=[],
        boxes=[],
        normalize_boxes=True,
        image_resolution=512,
        num_bins=1000,
):
    """
    method to create ReCo prompt

    caption: global caption
    phrases: list of regional captions
    boxes: list of regional coordinates (unnormalized xyxy)
    """

    SOS_token = '<|startoftext|>'
    EOS_token = '<|endoftext|>'

    box_captions_with_coords = []

    box_captions_with_coords += [caption]
    box_captions_with_coords += [EOS_token]

    for phrase, box in zip(phrases, boxes):

        if normalize_boxes:
            box = [float(x) / image_resolution for x in box]

        # quantize into bins
        quant_x0 = int(round((box[0] * (num_bins - 1))))
        quant_y0 = int(round((box[1] * (num_bins - 1))))
        quant_x1 = int(round((box[2] * (num_bins - 1))))
        quant_y1 = int(round((box[3] * (num_bins - 1))))

        # ReCo format
        # Add SOS/EOS before/after regional captions
        box_captions_with_coords += [
            f"<bin{str(quant_x0).zfill(3)}>",
            f"<bin{str(quant_y0).zfill(3)}>",
            f"<bin{str(quant_x1).zfill(3)}>",
            f"<bin{str(quant_y1).zfill(3)}>",
            SOS_token,
            phrase,
            EOS_token
        ]

    text = " ".join(box_captions_with_coords)
    return text


if __name__ == "__main__":
    pipe = StableDiffusionPipeline.from_pretrained(
        "j-min/reco_sd14_coco",
        torch_dtype=torch.float16
    )
    pipe = pipe.to("cuda")


    prompt = "A box contains six donuts with varying types of glazes and toppings. <|endoftext|> <bin514> <bin575> <bin741> <bin765> <|startoftext|> chocolate donut. <|endoftext|> <bin237> <bin517> <bin520> <bin784> <|startoftext|> dark vanilla donut. <|endoftext|> <bin763> <bin575> <bin988> <bin745> <|startoftext|> donut with sprinkles. <|endoftext|> <bin234> <bin281> <bin524> <bin527> <|startoftext|> donut with powdered sugar. <|endoftext|> <bin515> <bin259> <bin767> <bin514> <|startoftext|> pink donut. <|endoftext|> <bin753> <bin289> <bin958> <bin506> <|startoftext|> brown donut. <|endoftext|>"
    generated_image = pipe(
        prompt,
        guidance_scale=4
    ).images[0]



generated_image





caption = "a photo of bus and boat; boat is left to bus."
phrases = ["a photo of a bus.", "a photo of a boat."]
boxes = [[0.702, 0.404, 0.927, 0.601], [0.154, 0.383, 0.311, 0.487]]
prompt = create_reco_prompt(caption, phrases, boxes, normalize_boxes=False)
prompt
# >> > 'a photo of bus and boat; boat is left to bus. <|endoftext|> <bin701> <bin404> <bin926> <bin600> <|startoftext|> a photo of a bus. <|endoftext|> <bin154> <bin383> <bin311> <bin487> <|startoftext|> a photo of a boat. <|endoftext|>'

caption = "A box contains six donuts with varying types of glazes and toppings."
phrases = ["chocolate donut.", "dark vanilla donut.", "donut with sprinkles.", "donut with powdered sugar.",
           "pink donut.", "brown donut."]
boxes = [[263.68, 294.912, 380.544, 392.832], [121.344, 265.216, 267.392, 401.92], [391.168, 294.912, 506.368, 381.952],
         [120.064, 143.872, 268.8, 270.336], [264.192, 132.928, 393.216, 263.68], [386.048, 148.48, 490.688, 259.584]]
prompt = create_reco_prompt(caption, phrases, boxes)
prompt
# >> > 'A box contains six donuts with varying types of glazes and toppings. <|endoftext|> <bin514> <bin575> <bin743> <bin766> <|startoftext|> chocolate donut. <|endoftext|> <bin237> <bin517> <bin522> <bin784> <|startoftext|> dark vanilla donut. <|endoftext|> <bin763> <bin575> <bin988> <bin745> <|startoftext|> donut with sprinkles. <|endoftext|> <bin234> <bin281> <bin524> <bin527> <|startoftext|> donut with powdered sugar. <|endoftext|> <bin515> <bin259> <bin767> <bin514> <|startoftext|> pink donut. <|endoftext|> <bin753> <bin290> <bin957> <bin506> <|startoftext|> brown donut. <|endoftext|>'
