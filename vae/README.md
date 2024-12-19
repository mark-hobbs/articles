# Variational Autoencoders

1. **Data Generation**: 
   - Programmatically generate 2D shapes.

2. **Preprocessing**:
   - Normalise the shapes (e.g., scale them to a uniform size, centre them at the origin).
   - Convert the images or coordinate arrays into a consistent input format for your VAE, such as flattened vectors or tensors.

3. **VAE Implementation**:
   - Create a Variational Autoencoder using frameworks like TensorFlow or PyTorch.
   - The encoder compresses the high-dimensional input (shape data) into a low-dimensional latent space.
   - The decoder reconstructs the original shape from the latent space representation.

4. **Training**:
   - Train the VAE using your generated dataset. Use reconstruction loss (e.g., Mean Squared Error) and a KL-divergence term to regularise the latent space.

5. **Analysis**:
   - Visualise the latent space to understand the clustering or relationships between shapes.
   - Use the decoder to sample from the latent space and generate new shapes.

6. **Applications**:
   - Use the latent space for downstream tasks like shape classification, clustering, or generative design.