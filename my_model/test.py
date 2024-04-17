from model_predict import predict

model = predict()
image_path = '../dataset/val_images/0b1e7df6cc0e00d37ca7d16f14529304.jpg'
results = model.detect_image(image_path)
print(results)