import matplotlib.pyplot as plt

#thanks claude
def change_hex_brightness(hex_color: str, amount: float) -> str:
    """
    Lighten or darken a hex color by a specified amount.
    
    Args:
        hex_color (str): Hex color string (e.g., '#FF5733' or 'FF5733')
        amount (float): Amount to adjust (-1.0 to 1.0)
                       Positive values lighten, negative values darken
    
    Returns:
        str: New hex color string, maintaining original format (with or without '#')
    """
    # Check if original had '#' prefix
    has_hash: bool = hex_color.startswith('#')
    
    # Remove '#' if present
    hex_color = hex_color.lstrip('#')
    
    # Convert hex to RGB
    r: int = int(hex_color[0:2], 16)
    g: int = int(hex_color[2:4], 16)
    b: int = int(hex_color[4:6], 16)
    
    # Adjust each color component
    if amount > 0:  # Lighten
        r = int(r + (255 - r) * amount)
        g = int(g + (255 - g) * amount)
        b = int(b + (255 - b) * amount)
    else:  # Darken
        r = int(r * (1 + amount))
        g = int(g * (1 + amount))
        b = int(b * (1 + amount))
    
    # Ensure values stay within 0-255 range
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))
    
    # Convert back to hex, preserving original format
    hex_result: str = f"{r:02x}{g:02x}{b:02x}"
    return f"#{hex_result}" if has_hash else hex_result


def get_sample_mapping():
    sample_mapping = {
        '216':'Ovarian A',
        '244':'Ovarian B',
        '264':'Ovarian C',
        '053':'Ovarian D',
        #'192':'Ovarian E',
        'BS14772':"Prostate A",
        'BS15145':'Prostate B',
    }

    return sample_mapping
def get_sample_color_scheme():
    sample_color_scheme = {
        '216':'#ff2e1f',
        '244':'#ff8000',
        '264':'#ffbf00',
        '053':'#85221e',
        'BS14772':"#8ecae6",
        'BS15145':'#219ebc',
    }

    sample_color_scheme_temp = dict(sample_color_scheme)
    for mode in ['NL','TU']:
        for sample,color in sample_color_scheme_temp.items():
            sample_color_scheme[f'{sample}_{mode}'] = color
    del sample_color_scheme_temp
    
    sample_mapping = get_sample_mapping()

    for original_sample,mapped_sample in sample_mapping.items():
        sample_color_scheme[mapped_sample] = sample_color_scheme[original_sample]
    return sample_color_scheme

sample_mapping = get_sample_mapping()
sample_color_scheme = get_sample_color_scheme()


def setup_plot_style():

    # Font settings
    plt.rcParams.update({
        'pdf.fonttype': 42,
        'font.family': 'sans-serif',
    })
# Automatically apply settings when imported
setup_plot_style()

