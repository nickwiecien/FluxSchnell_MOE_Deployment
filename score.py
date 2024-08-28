import pandas as pd
import json
import os
from azureml.ai.monitoring import Collector
import torch
from diffusers import FluxPipeline
import base64
from io import BytesIO

def init():
    global inputs_collector, outputs_collector, inputs_outputs_collector, loaded_model
    loaded_model = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16)
    loaded_model.to("cuda")
    inputs_collector = Collector(name='model_inputs')                    
    outputs_collector = Collector(name='model_outputs')
    
def run(data):
    print(data)
    print(type(data))
    data = json.loads(data)
    prompt = data['prompt']
    inference_steps = int(data['inference_steps'])
    guidance = float(data['guidance_scale'])
    
    # Create a DataFrame to collect input data
    input_df = pd.DataFrame([data])
    inputs_collector.collect(input_df)
    
    # Generate the image using the model
    image = loaded_model(
        prompt,
        guidance_scale=guidance,
        output_type="pil",
        num_inference_steps=inference_steps,
        max_sequence_length=512,
        generator=torch.Generator("cuda"),
    ).images[0]
    
    # Convert the PIL image to a base64 string
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    # Save the base64 string to a DataFrame
    output_df = pd.DataFrame({
        "prompt": [prompt],
        "image_base64": [image_base64]
    })
    
    outputs_collector.collect(output_df)
    
    # Return the image as a base64 string in the response
    return {"image_base64": image_base64}
 
    