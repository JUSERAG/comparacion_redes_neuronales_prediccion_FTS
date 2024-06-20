from tkinter import filedialog
from typing import List, Optional
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

# Definir la función show_metrics_table()
def show_metrics_table():
    # Crear ventana emergente para mostrar la tabla de métricas
    root = tk.Tk()
    root.title("Tabla de Métricas de Rendimiento")

    # Crear un Frame para el Treeview
    frame = ttk.Frame(root)
    frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

    # Crear Treeview
    tree = ttk.Treeview(frame, columns=list(metrics_df.columns), show='headings')
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    # Configurar encabezados
    for col in metrics_df.columns:
        tree.heading(col, text=col)
        tree.column(col, width=100)

    # Agregar datos
    for index, row in metrics_df.iterrows():
        formatted_row = [
            f"{value:.4f}" if isinstance(value, (int, float)) and col != 'MAPE' else 
            (f"{value:.2f}%" if col == 'MAPE' else value) 
            for col, value in zip(metrics_df.columns, row)
        ]
        tree.insert("", tk.END, values=formatted_row)

    # Crear un Scrollbar
    scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    tree.config(yscrollcommand=scrollbar.set)

    # Función para exportar la tabla como un archivo PDF
    def export_table_to_pdf():
        file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
        if file_path:
            with PdfPages(file_path) as pdf:
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111)
                ax.axis('tight')
                ax.axis('off')
                # Formatear los datos de la tabla para el PDF
                table_data = [
                    [f"{value:.4f}" if isinstance(value, (int, float)) and col != 'MAPE' else 
                     (f"{value:.2f}%" if col == 'MAPE' else value) 
                     for col, value in zip(metrics_df.columns, row)]
                    for _, row in metrics_df.iterrows()
                ]
                ax.table(cellText=table_data, colLabels=metrics_df.columns, loc='center')
                pdf.savefig(fig, bbox_inches='tight')

    # Botón para exportar la tabla como PDF
    export_button = ttk.Button(root, text="Exportar a PDF", command=export_table_to_pdf)
    export_button.pack(pady=10)

    # Iniciar loop de la ventana
    root.mainloop()

# Definir los activos financieros
assets = {
    "Indices": ["^GSPC", "^N225", "^DJI", "^FTSE", "^GDAXI", "^IXIC", "^HSI", "^NYA", "000001.SS", "^NSEI", "^RUT", "^HSI"],
    "Acciones": ["AAPL", "AMZN", "INTC", "T", "BAC", "NFLX", "TSLA", "META", "JPM", "MSFT", "EBAY", "GOOGL"],
    "Materias Primas": ["GC=F", "ZS=F", "CL=F"],
    "Divisas": ["EURUSD=X", "JPYUSD=X", "GBPUSD=X", "INRUSD=X", "SGDUSD=X", "USDCNY=X", "EURCNY=X", "JPYCNY=X", "CHFCNY=X", "AUDCAD=X"],
    "Criptomonedas": ["BTC-USD", "LTC-USD", "ETH-USD", "ZEC-USD", "XLM-USD", "XRP-USD"]
}

# Función para descargar datos de Yahoo Finance
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Crear características de series temporales (ventana deslizante)
def create_features(prices, window_size):
    x, y = [], []
    for i in range(len(prices) - window_size):
        x.append(prices[i:i + window_size])
        y.append(prices[i + window_size])
    return np.array(x), np.array(y)

# Modelo Dendritic Neural Network
def step_square_loss(inputs: np.ndarray,
                     weights: List[np.ndarray],
                     hyperplanes: List[np.ndarray],
                     hyperplane_bias_magnitude: Optional[float] = 1.,
                     learning_rate: Optional[float] = 1e-5,
                     target: Optional[float] = None,
                     update: bool = False):
    r_in = inputs
    side_info = np.hstack([hyperplane_bias_magnitude, inputs])

    for w, h in zip(weights, hyperplanes):  # loop over layers
        r_in = np.hstack([1., r_in])  # add biases
        gate_values = np.heaviside(h.dot(side_info), 0).astype(bool)
        effective_weights = gate_values.dot(w).sum(axis=1)
        r_out = effective_weights.dot(r_in)

        if update:
            grad = (r_out[:, None] - target) * r_in[None]
            w -= learning_rate * gate_values[:, :, None] * grad[:, None]

        r_in = r_out
    loss = (target - r_out)**2 / 2
    return r_out, loss

def forward_pass(step_fn, x, y, weights, hyperplanes, learning_rate, update):
    losses, outputs = np.zeros(len(y)), np.zeros(len(y))
    for i, (x_i, y_i) in enumerate(zip(x, y)):
        outputs[i], losses[i] = step_fn(x_i, weights, hyperplanes, target=y_i,
                                        learning_rate=learning_rate, update=update)
    return np.mean(losses), outputs

# Configuración del modelo
n_neurons = np.array([100, 10, 1])
n_branches = 20  # número de ramas dendríticas por neurona
eta = 1e-5
n_epochs = 10

# Almacenar todas las gráficas y métricas temporalmente
all_plots = []
metric_results = []
start_date = '2015-01-01'
end_date = '2019-12-31'

# Procesar y entrenar el modelo para cada activo
for category, tickers in assets.items():
    for ticker in tickers:
        print(f'Processing {ticker}...')

        # Descargar datos
        data = download_data(ticker, start_date, end_date)

        # Usar la columna 'Close' como el objetivo a predecir
        prices = data['Close'].values
        dates = data.index  # Fechas correspondientes a los datos

        # Crear características y objetivos
        features, targets = create_features(prices, window_size=30)

        # Dividir los datos en conjuntos de entrenamiento y prueba
        x_train, x_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=0)
        n_features = x_train.shape[-1]

        # Escalar las características
        feature_encoder = preprocessing.StandardScaler()
        x_train = feature_encoder.fit_transform(x_train)
        x_test = feature_encoder.transform(x_test)

        # Escalar los objetivos
        target_encoder = preprocessing.StandardScaler()
        y_train = np.squeeze(target_encoder.fit_transform(y_train[:, np.newaxis]))
        y_test = np.squeeze(target_encoder.transform(y_test[:, np.newaxis]))

        # Inicializar pesos y hiperplanos
        n_inputs = np.hstack([n_features + 1, n_neurons[:-1] + 1])  # 1 para el sesgo
        dgn_weights = [np.zeros((n_neuron, n_branches, n_input))
                       for n_neuron, n_input in zip(n_neurons, n_inputs)]
        np.random.seed(12345)
        dgn_hyperplanes = [
            np.random.normal(0, 1, size=(n_neuron, n_branches, n_features + 1))
            for n_neuron in n_neurons]
        dgn_hyperplanes = [
            h_ / np.linalg.norm(h_[:, :, :-1], axis=(1, 2))[:, None, None]
            for h_ in dgn_hyperplanes]

        # Entrenamiento del modelo
        for epoch in range(0, n_epochs + 1):
            train_loss, train_pred = forward_pass(
                step_square_loss, x_train, y_train, dgn_weights,
                dgn_hyperplanes, eta, update=(epoch > 0))

            test_loss, test_pred = forward_pass(
                step_square_loss, x_test, y_test, dgn_weights,
                dgn_hyperplanes, eta, update=False)
            to_print = 'epoch: {}, test loss: {:.3f} (train: {:.3f})'.format(
                epoch, test_loss, train_loss)

            print(to_print)

        # Invertir la escala de las predicciones
        train_pred = train_pred.reshape(-1, 1)
        test_pred = test_pred.reshape(-1, 1)
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)

        train_pred = target_encoder.inverse_transform(train_pred)
        test_pred = target_encoder.inverse_transform(test_pred)
        y_train = target_encoder.inverse_transform(y_train)
        y_test = target_encoder.inverse_transform(y_test)

        # Calcular métricas de rendimiento
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mape = np.mean(np.abs((y_test - test_pred) / y_test)) * 100
        test_mae = mean_absolute_error(y_test, test_pred)
        test_mse = mean_squared_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)

        # Almacenar resultados
        metric_results.append([ticker, test_mse, test_rmse, test_mae, test_mape, test_r2])

        # Almacenar gráficas
        all_plots.append((y_test, test_pred, dates, ticker))

# Crear DataFrame para las métricas de rendimiento
metrics_df = pd.DataFrame(metric_results, columns=[
    'Activo', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2'
])


# Diccionario para la configuración de la fuente
fontdict = {'family': 'serif', 'weight': 'bold'}

# Mostrar todas las gráficas juntas al final
plt.figure(figsize=(20, 5))
plt.suptitle(f"Red DNM: datos tomados desde {start_date} hasta {end_date}", fontsize=10)

# Ajustar espaciado entre gráficos
plt.subplots_adjust(hspace=1.5)

for idx, (y_test, test_pred, dates, ticker) in enumerate(all_plots, start=1):
    plt.subplot((len(all_plots) + 2) // 3, 3, idx)
    plt.plot(dates[len(dates) - len(y_test):], y_test, linewidth=0.8)
    plt.plot(dates[len(dates) - len(test_pred):], test_pred, linewidth=0.6)
    plt.title(ticker, fontsize='small', fontdict=fontdict)
    plt.xticks([])
    plt.yticks([])
plt.tight_layout()
plt.show()

# Llamar a la función show_metrics_table() al final
show_metrics_table()
