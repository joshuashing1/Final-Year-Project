import numpy as np
import pandas as pd
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# --- Helper function to parse maturities in years ---
def parse_maturities(maturities):
    parsed = []
    for m in maturities:
        m = str(m).strip().replace(' ', '')
        if 'M' in m or 'Mo' in m:
            number = float(''.join([ch for ch in m if ch.isdigit() or ch == '.']))
            parsed.append(number / 12)
        elif 'Y' in m or 'Yr' in m:
            number = float(''.join([ch for ch in m if ch.isdigit() or ch == '.']))
            parsed.append(number)
        else:
            parsed.append(float(m))
    return np.array(parsed)

# --- Path and Data Loading ---
csv_path = 'Chapter 2\Data\GBP-Yield-Curve.csv'   # Change as needed
country = "GBP"                                   # Change as needed
df = pd.read_csv(csv_path)
maturities = [str(c).strip() for c in df.columns]
T_years = parse_maturities(maturities)
X = df.values  # shape: (num_samples, num_maturities)

# --- Autoencoder Construction ---
input_dim = X.shape[1]
encoding_dim = 13  # You can tune this

input_yield = Input(shape=(input_dim,))
encoded = Dense(30, activation='relu')(input_yield)
encoded = Dense(encoding_dim, activation='relu')(encoded)

decoded = Dense(30, activation='relu')(encoded)
decoded = Dense(input_dim, activation='linear')(decoded)

autoencoder = Model(input_yield, decoded)
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.fit(X, X, epochs=200, batch_size=8, verbose=1)

# --- Encoder and Decoder ---
encoder = Model(input_yield, encoded)
encoded_input = Input(shape=(encoding_dim,))
decoder_layer1 = autoencoder.layers[-2](encoded_input)
decoder_layer2 = autoencoder.layers[-1](decoder_layer1)
decoder = Model(encoded_input, decoder_layer2)

encoded_yields = encoder.predict(X)
decoded_yields = autoencoder.predict(X)

# --- Calculate RMSE for all curves ---
rmses = np.sqrt(np.mean((X - decoded_yields) ** 2, axis=1))
avg_rmse = np.mean(rmses)

# --- Plot all reconstructed curves in a single plot ---
fig, ax = plt.subplots(figsize=(12, 6))
for i in range(decoded_yields.shape[0]):
    ax.plot(T_years, decoded_yields[i], lw=1)  # Each reconstructed curve

# Title and axis formatting
ax.set_title(country, fontsize=40, weight='bold', color='#183057', pad=10)
ax.set_xlabel("Maturity (years)", fontsize=25, color='#183057', weight='bold')
ax.set_ylabel("Swap Rate (%)", fontsize=25, color='#183057', weight='bold')
ax.set_xlim([min(T_years), max(T_years)])
ax.set_ylim(-2, 11)
ax.tick_params(axis='both', colors='#183057', labelsize=18)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontsize(22)
    label.set_color('#183057')
for spine in ['left', 'bottom']:
    ax.spines[spine].set_color('#183057')
    ax.spines[spine].set_linewidth(2)
for spine in ['top', 'right']:
    ax.spines[spine].set_visible(False)

# Average RMSE text box (upper right)
textstr = f"Avg. RMSE = {avg_rmse:.4f}"
ax.text(
    0.98, 0.98, textstr,
    transform=ax.transAxes,
    fontsize=22,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(facecolor='white', edgecolor='red', boxstyle='square,pad=0.4', alpha=0.85)
)

plt.tight_layout()
plt.show()
