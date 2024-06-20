import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error, f1_score
import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Definición de la TCN
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.final_layer = nn.Linear(num_channels[-1], num_inputs)

    def forward(self, x):
        y = self.network(x)
        y = y[:, :, -1]  # Solo tomamos el último punto de la secuencia
        return self.final_layer(y)

# Descargar datos financieros
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)['Close']
    return data

# Preparar datos para TCN
def prepare_data(data, window_size):
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data.values.reshape(-1, 1))
    
    x = []
    y = []
    for i in range(window_size, len(data_scaled)):
        x.append(data_scaled[i-window_size:i])
        y.append(data_scaled[i])
    
    x = np.array(x)
    y = np.array(y)
    
    return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32), scaler

# Entrenar el modelo
def train_model(model, train_data, train_labels, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        output = model(train_data)
        loss = criterion(output, train_labels)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluar el modelo
def evaluate_model(model, test_data, test_labels, scaler):
    model.eval()
    with torch.no_grad():
        predictions = model(test_data).numpy()

    predictions = scaler.inverse_transform(predictions)
    true_values = scaler.inverse_transform(test_labels.numpy())

    mse = mean_squared_error(true_values, predictions)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_values, predictions)
    mape = mean_absolute_percentage_error(true_values, predictions) * 100  # Convertir a porcentaje
    mae = mean_absolute_error(true_values, predictions)

    return true_values, predictions, mse, rmse, mae, mape, r2

# Parámetros
start_date = '2015-01-01'
end_date = '2019-12-31'
window_size = 60
num_channels = [25, 50, 100]
num_epochs = 1
learning_rate = 0.001

# Activos financieros
assets = {
    "Indices": ["^GSPC", "^N225", "^DJI", "^FTSE", "^GDAXI", "^IXIC", "^HSI", "^NYA", "000001.SS", "^NSEI", "^RUT", "^HSI"],
    "Acciones": ["AAPL", "AMZN", "INTC", "T", "BAC", "NFLX", "TSLA", "META", "JPM", "MSFT", "EBAY", "GOOGL"],
    "Materias Primas": ["GC=F", "ZS=F", "CL=F"],
    "Divisas": ["EURUSD=X", "JPYUSD=X", "GBPUSD=X", "INRUSD=X", "SGDUSD=X", "USDCNY=X", "EURCNY=X", "JPYCNY=X", "CHFCNY=X", "AUDCAD=X"],
    "Criptomonedas": ["BTC-USD", "LTC-USD", "ETH-USD", "ZEC-USD", "XLM-USD", "XRP-USD"]
}

# Listas para guardar resultados
results = []

# Iterar sobre cada activo financiero
for category, tickers in assets.items():
    for ticker in tickers:
        print(f"Processing {ticker}...")

        # Descargar y preparar datos
        data = download_data(ticker, start_date, end_date)
        if data is None or data.empty:
            print(f"No data for {ticker}")
            continue

        train_data, train_labels, scaler = prepare_data(data, window_size)

        # Ajustar dimensiones para TCN
        train_data = train_data.permute(0, 2, 1)  # Cambiar dimensiones a (batch_size, num_features, seq_len)

        # Definir el modelo
        model = TemporalConvNet(num_inputs=1, num_channels=num_channels)  # num_inputs=1 porque solo estamos usando el precio de cierre

        # Entrenar el modelo
        train_model(model, train_data, train_labels, num_epochs, learning_rate)

        # Evaluar el modelo
        true_values, predictions, mse, rmse, mae, mape, r2 = evaluate_model(model, train_data, train_labels, scaler)
        
        # Guardar resultados
        results.append((ticker, data.index[-len(true_values):], true_values, predictions, mse, rmse, mae, mape, r2))

# Diccionario para la configuración de la fuente
fontdict = {'family': 'serif', 'weight': 'bold'}

# Graficar todos los resultados en subplots
num_assets = len(results)
rows = (num_assets // 3) + (1 if num_assets % 3 > 0 else 0)
fig, axs = plt.subplots(rows, 3, figsize=(20, 5 * rows))
fig.suptitle(f"Red TCN: datos tomados desde {start_date} hasta {end_date}", fontsize=10)
fig.tight_layout(pad=5.0, rect=[0, 0.03, 1, 0.95])

# Ajustar espaciado entre gráficos
plt.subplots_adjust(hspace=1.5)

for i, (ticker, dates, true_values, predictions, mse, rmse, mae, mape, r2) in enumerate(results):
    ax = axs[i // 3, i % 3] if rows > 1 else axs[i % 3]
    ax.plot(dates, true_values)
    ax.plot(dates, predictions)
    ax.set_title(ticker, fontsize='small', fontdict=fontdict)
    ax.set_xticks([])  # Eliminar etiquetas de fechas en el eje x
    ax.set_yticks([])  # Eliminar etiquetas del eje y

# Ocultar subplots vacíos
for j in range(i + 1, rows * 3):
    if rows > 1:
        fig.delaxes(axs[j // 3, j % 3])
    else:
        fig.delaxes(axs[j % 3])

plt.show()

# Crear la tabla de métricas
metrics_table = pd.DataFrame(
    [(ticker, f"{mse:.4f}", f"{rmse:.4f}", f"{mae:.4f}", f"{mape:.2f}%", f"{r2:.4f}") for ticker, _, _, _, mse, rmse, mae, mape, r2 in results],
    columns=['Activo', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2']
)

# Función para exportar la tabla como un archivo PDF
def export_table():
    file_path = filedialog.asksaveasfilename(defaultextension=".pdf", filetypes=[("PDF files", "*.pdf")])
    if file_path:
        with PdfPages(file_path) as pdf:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.axis('tight')
            ax.axis('off')
            ax.table(cellText=metrics_table.values, colLabels=metrics_table.columns, loc='center')
            pdf.savefig(fig, bbox_inches='tight')

# Crear ventana
root = tk.Tk()
root.title("Tabla de Métricas de Rendimiento")

# Crear un Frame para el Treeview
frame = ttk.Frame(root)
frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)

# Crear Treeview
tree = ttk.Treeview(frame, columns=list(metrics_table.columns), show='headings')
tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

# Configurar encabezados
for col in metrics_table.columns:
    tree.heading(col, text=col)
    tree.column(col, width=100)

# Agregar datos
for index, row in metrics_table.iterrows():
    tree.insert("", tk.END, values=list(row))

# Crear un Scrollbar
scrollbar = ttk.Scrollbar(frame, orient=tk.VERTICAL, command=tree.yview)
scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
tree.config(yscrollcommand=scrollbar.set)

# Botón para exportar la tabla
export_button = ttk.Button(root, text="Exportar Tabla", command=export_table)
export_button.pack(pady=10)

# Iniciar loop de la ventana
root.mainloop()
