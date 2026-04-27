# Models

Trained model weights are excluded from version control (see `.gitignore`).

To save models after training, add these calls to the notebook or `src/` scripts:

```python
# Save LSTM forecaster
lstm_model.save("models/lstm_forecaster.keras")

# Save irrigation classifier
final_model.save("models/irrigation_classifier.keras")
```

To reload:

```python
from tensorflow.keras.models import load_model

lstm_model = load_model("models/lstm_forecaster.keras")
final_model = load_model("models/irrigation_classifier.keras")
```
