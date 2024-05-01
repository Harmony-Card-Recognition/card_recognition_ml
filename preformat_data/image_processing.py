# we can also have the image be cropped so that it is a true rectangle (kind of like what genus scan does with their documents)
# this will help us make sure all of the data is in the right spot, (and that it isn't rotated or something like that)


# this will handle all of the image processing that will make the neural network have an easier time figuring out what is on the card
    # some ideas include high contrasting the images
    # or removing some of the glare
    # or converting it to greyscale

# you could also do something like make it so that the images are all the same size
# or crop some parts of the images, so that the model knows what to look for where 
    # since we always know that the title will be at the top, and the health will be at the upper right corner
    # or that the picture is basically irreleant... maybe it is, maybe not
    # either way, we can divide the card into something better