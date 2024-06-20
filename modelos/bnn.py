import torch
import torch.nn as nn
import torch.optim as optim
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error, mean_absolute_error
from torch.utils.data import Dataset, DataLoader, random_split
from matplotlib.backends.backend_pdf import PdfPages
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
from tkinter.filedialog import asksaveasfilename
from tkinter import filedialog

# Descargar datos financieros
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)['Close']
    return data

# Preparar datos para BNN
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

class FinancialDataset(Dataset):
    def __init__(self, data, target):
        self.data = data
        self.target = target

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.target[idx]

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mu = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-0.2, 0.2))
        self.rho = nn.Parameter(torch.Tensor(out_features, in_features).uniform_(-5, -4))
        self.register_buffer('prior_mu', torch.Tensor(out_features, in_features).fill_(0))
        self.register_buffer('prior_sigma', torch.Tensor(out_features, in_features).fill_(1))
        self.register_buffer('prior_log_sigma', torch.log(self.prior_sigma))

    def forward(self, input):
        sigma = torch.log1p(torch.exp(self.rho))
        epsilon = torch.randn_like(sigma)
        weight = self.mu + sigma * epsilon
        return nn.functional.linear(input, weight)

class BNN_Regression(nn.Module):
    def __init__(self, params):
        super(BNN_Regression, self).__init__()
        self.fc1 = BayesianLinear(params['x_shape'], params['hidden_units'])
        self.fc2 = BayesianLinear(params['hidden_units'], params['y_shape'])

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Asegurarse de que x tenga la forma correcta
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

def train_model(model, train_loader, num_epochs=100, learning_rate=0.001):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}')

def evaluate_model(model, test_loader, scaler):
    model.eval()
    predictions = []
    true_values = []
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            output = model(x_batch)
            predictions.append(output.numpy())
            true_values.append(y_batch.numpy())

    predictions = scaler.inverse_transform(np.vstack(predictions))
    true_values = scaler.inverse_transform(np.vstack(true_values))

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
hidden_units = 400
num_epochs = 200
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

        data_tensor, target_tensor, scaler = prepare_data(data, window_size)

        dataset = FinancialDataset(data_tensor, target_tensor)
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

        # Definir el modelo
        params = {
            'x_shape': window_size,
            'y_shape': 1,
            'hidden_units': hidden_units,
        }
        model = BNN_Regression(params)

        # Entrenar el modelo
        train_model(model, train_loader, num_epochs, learning_rate)

        # Evaluar el modelo
        true_values, predictions, mse, rmse, mae, mape, r2 = evaluate_model(model, test_loader, scaler)
        
        # Guardar resultados
        results.append((ticker, data.index[-len(true_values):], true_values, predictions, mse, rmse, mae, mape, r2))

# Diccionario para la configuración de la fuente
fontdict = {'family': 'serif', 'weight': 'bold'}

# Graficar todos los resultados en subplots
num_assets = len(results)
rows = (num_assets // 3) + (1 if num_assets % 3 > 0 else 0)
fig, axs = plt.subplots(rows, 3, figsize=(20, 5 * rows))
fig.suptitle(f"Red BNN: datos tomados desde {start_date} hasta {end_date}", fontsize=10)
fig.tight_layout(pad=5.0, rect=[0, 0.03, 1, 0.95])

# Ajustar espaciado entre gráficos
plt.subplots_adjust(hspace=1.5)

for i, (ticker, dates, true_values, predictions, mse, rmse, mae, mape, r2) in enumerate(results):
    ax = axs[i // 3, i % 3] if rows > 1 else axs[i % 3]
    ax.plot(dates, true_values, linewidth=0.8)
    ax.plot(dates, predictions, linewidth=0.7)
    ax.set_title(ticker, fontsize='small', fontdict=fontdict)
    ax.set_xticks([])  # Eliminar etiquetas de fechas en el eje x
    ax.set_yticks([])  # Eliminar etiquetas del eje y

# Ocultar subplots vacíos
for j in range(i + 1, rows * 3):
    if rows > 1:
        fig.delaxes(axs[j // 3, j % 3])
    else:
        fig.delaxes(axs[j % 3])

# Mostrar la gráfica
plt.show()
        
# Crear tabla de métricas
metrics_table = pd.DataFrame(results, columns=['Activo', 'Dates', 'True Values', 'Predictions', 'MSE', 'RMSE', 'MAE', 'MAPE', 'R2'])
metrics_table = metrics_table.drop(['Dates', 'True Values', 'Predictions'], axis=1)
metrics_table['MSE'] = metrics_table['MSE'].round(4)
metrics_table['RMSE'] = metrics_table['RMSE'].round(4)
metrics_table['MAE'] = metrics_table['MAE'].round(4)
metrics_table['MAPE'] = metrics_table['MAPE'].round(2)
metrics_table['R2'] = metrics_table['R2'].round(4)



# Función para exportar la tabla como un archivo PDF
def export_to_pdf():
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
                    for col, value in zip(metrics_table.columns, row)]
                for _, row in metrics_table.iterrows()
            ]
            ax.table(cellText=table_data, colLabels=metrics_table.columns, loc='center')
            pdf.savefig(fig, bbox_inches='tight')
            
# Ventana principal de tkinter
root = tk.Tk()
root.title("Métricas de Rendimiento")

# Crear tabla en tkinter
frame = ttk.Frame(root)
frame.pack(padx=10, pady=10)
tree = ttk.Treeview(frame, columns=list(metrics_table.columns), show='headings', height=15)
tree.pack(side='left')

# Configurar encabezados de la tabla
for col in metrics_table.columns:
    tree.heading(col, text=col)
    tree.column(col, anchor="center")

# Agregar filas a la tabla
for index, row in metrics_table.iterrows():
    tree.insert("", "end", values=list(row))

# Agregar scrollbar
scrollbar = ttk.Scrollbar(frame, orient='vertical', command=tree.yview)
scrollbar.pack(side='right', fill='y')
tree.configure(yscroll=scrollbar.set)

# Botón para exportar a PDF
export_button = ttk.Button(root, text="Exportar a PDF", command=export_to_pdf)
export_button.pack(pady=10)

root.mainloop()
