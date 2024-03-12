`colors.py` - a store of colours for background removal and segmentation \
`custom_nodes.py` - modification to certain helper methods that access the ComfyUI library. \
`general_config.toml` - base config for training. \
`general_dataset.toml` - base config for dataset settings. \
`generate_paintings.py` - use ComfyUI to generate paintings via prompt and trained model. \
`one_pass.py` - takes request and starts a chain of commands to remove backgrounds, train face lora, generate paintings and them back. \
`preproc.py` - to process incoming face images and remove background. \
`server_utils.py` - helper methods for server io stuff. \
`test_server.py` - test code. \
`upscale.py` - upscale pipeline. \
`worker_server -` flask code for running a server with state. \
