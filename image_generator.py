import replicate
import os
import requests

# Set your API token
os.environ["REPLICATE_API_TOKEN"] = "r8_eAUWS3mpjgbcGxeJNShTmPHMvyMj2jd3jcCy4"




def furnish_image(img_file_name, image_url, style, room_type):

    prompt = f"A stylish {style} {room_type} Featuring a sofa,  coffee table, rug, and floor lamps. warm ambient lighting. Photorealistic, cinematic lighting, 8k"


    # Run the model
    output = replicate.run(
        "jschoormans/comfyui-interior-remodel:2a360362540e1f6cfe59c9db4aa8aa9059233d40e638aae0cdeb6b41f3d0dcce",
        input={
            "image": image_url,
            "prompt": prompt,
            "output_quality": 80,
            "output_format": "png"
        }
    )

    # Save the result(s)
    for index, item in enumerate(output):
        image_path = f"static/generated_images/output_{img_file_name}_{index}.png"
        with open(image_path, "wb") as f:
            f.write(item.read())
        print(f"✅ Saved: output_{index}.png")

    return image_path
