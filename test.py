import firefly

x = firefly.Client("https://style-transfer-demo.rorocloud.io/")

new_img = x.style_transfer(
    content_fileobj=open("data/two_birds_photo.jpg", "rb"),
    style_fileobj=open("data/picasso.jpg", "rb"),
    n_iterations=1)

with open("new.jpg", "wb") as f:
    f.write(new_img.read())

