"""Image preprocessing helpers that do not depend on the training stack."""

from PIL import Image


def add_margins_to_image(img, margin_size):
    """Add mirrored margins to an RGB image."""
    margin_left = img.crop((0, 0, margin_size, img.height)).transpose(Image.FLIP_LEFT_RIGHT)
    margin_right = img.crop((img.width - margin_size, 0, img.width, img.height)).transpose(Image.FLIP_LEFT_RIGHT)
    margin_top = img.crop((0, 0, img.width, margin_size)).transpose(Image.FLIP_TOP_BOTTOM)
    margin_bottom = img.crop((0, img.height - margin_size, img.width, img.height)).transpose(Image.FLIP_TOP_BOTTOM)

    margin_top_left = img.crop((0, 0, margin_size, margin_size)).transpose(Image.FLIP_LEFT_RIGHT).transpose(
        Image.FLIP_TOP_BOTTOM)
    margin_top_right = img.crop((img.width - margin_size, 0, img.width, margin_size)).transpose(
        Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    margin_bottom_left = img.crop((0, img.height - margin_size, margin_size, img.height)).transpose(
        Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)
    margin_bottom_right = img.crop(
        (img.width - margin_size, img.height - margin_size, img.width, img.height)).transpose(
        Image.FLIP_LEFT_RIGHT).transpose(Image.FLIP_TOP_BOTTOM)

    concat_img = Image.new('RGB', (img.width + margin_size * 2, img.height + margin_size * 2))

    concat_img.paste(img, (margin_size, margin_size))
    concat_img.paste(margin_left, (0, margin_size))
    concat_img.paste(margin_right, (img.width + margin_size, margin_size))
    concat_img.paste(margin_top, (margin_size, 0))
    concat_img.paste(margin_bottom, (margin_size, img.height + margin_size))
    concat_img.paste(margin_top_left, (0, 0))
    concat_img.paste(margin_top_right, (img.width + margin_size, 0))
    concat_img.paste(margin_bottom_left, (0, img.height + margin_size))
    concat_img.paste(margin_bottom_right, (img.width + margin_size, img.height + margin_size))

    return concat_img
