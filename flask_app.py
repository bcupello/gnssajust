from flask import Flask
from flask import render_template
from flask import request, redirect, url_for, send_file
from flask_table import Table, Col
import os
import csv
import pandas as pd
import numpy as np
from scipy.stats import chi2
app = Flask(__name__)

ALLOWED_EXTENSIONS = set(['csv'])
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
APP_STATIC = os.path.join(APP_ROOT, 'static')

# Declara a tabela de pontos ajustados
class Adjusted_Pairs_Table(Table):
    ponto = Col('Ponto')
    x = Col('X')
    y = Col('Y')
    z = Col('Z')


# Declara a classe dos pontos ajustados
class Adjusted_Pair(object):
    def __init__(self, ponto, x, y, z):
        self.ponto = ponto
        self.x = x
        self.y = y
        self.z = z

def readcsv(filename):	
    ifile = open(filename, "rU")
    reader = csv.reader(ifile, delimiter=";")

    rownum = 0	
    a = []

    for row in reader:
        a.append (row)
        rownum += 1
    
    ifile.close()
    return a

def calc_answer(mat_obs, mat_pts_ctrl):
	# Faz as contas

	# Retorna a resposta
	return None

@app.route('/ajuste', methods=['GET', 'POST'])
def ajuste():
    if request.method == 'GET':
        return render_template('ajuste.html')
    if request.method == 'POST':
        # check if the post request has the file part
        if 'csv_file' not in request.files:
        	print('No file part')
        	return redirect(request.url)
        df = pd.read_csv(request.files.get('csv_file'))
        
        # Traduz o arquivo em um array de arrays
        matrix = df.reset_index().values

        arrays_matrix = []
        for i in range(len(matrix)):
            arrays_matrix.append(matrix[i][1].split(';'))

        # Coloca os dados no formato da função que resolve o problema matemático
        observation_matrix = []
        std_patterns_matrix = []
        ctrl_matrix = []
        var_a_priori = 1
        for obs in arrays_matrix:
            # Preenche a matriz de observações
            observation_matrix.append([int(obs[0]),int(obs[1]),float(obs[2]),float(obs[4]),float(obs[6])])
            std_patterns_matrix.append([float(obs[3]),float(obs[5]),float(obs[7])])
            # Preenche a matriz de pontos de controle
            if(obs[8] != ''):
                ctrl_matrix.append([int(obs[8]),float(obs[9]),float(obs[10]),float(obs[11])])
        # Dá o valor da variância a priori
        if(arrays_matrix[0][12] != ''):
            var_a_priori = float(arrays_matrix[0][12])
        
        # Chama a função que resolve o problema matemático e obtém o resultado
        M_par_aj, MVC_final, M_elipsoides_de_erro, teste_hipotese_aceito = calc_ajuste_rede(observation_matrix, std_patterns_matrix, ctrl_matrix, var_a_priori)

        # Prepara a tabela de VC
        M_VC = []
        for vc_line in MVC_final:
            new_vc_line = []
            for vc in vc_line:
                new_vc_line.append(round(vc, 6))
            M_VC.append(new_vc_line.copy())

        # Prepara a tabela de pontos ajustados
        adjusted_pairs = []
        for adj_pair in M_par_aj:
            adjusted_pairs.append(Adjusted_Pair(adj_pair[0], round(adj_pair[1], 6), round(adj_pair[2], 6), round(adj_pair[3], 6)))

        adjusted_pairs_table = Adjusted_Pairs_Table(adjusted_pairs)

        # Prepara as elipsoides de erro
        M_EdE = []
        for ede_line in M_elipsoides_de_erro:
            new_ede_line = []
            for ede in ede_line:
                new_ede_line.append(round(ede, 6))
            M_EdE.append(new_ede_line.copy())

        # Retorna para a página os dados carregados
        return render_template('ajuste.html', Adjusted_pairs_table=adjusted_pairs_table, M_VC=M_VC, M_EdE=M_EdE, Teste_hipotese_aceito=teste_hipotese_aceito)

@app.route('/exemplo')
def exemplo():
	return render_template('exemplo.html')

@app.route('/')
def index():
	return render_template('index.html')

@app.route('/codigo')
def codigo():
	return render_template('codigo.html')

@app.route('/example-file/')
def return_example_file():
    try:
        return send_file(os.path.join(APP_STATIC, 'example.csv'), attachment_filename='example.csv')
    except Exception as e:
        return str(e)

@app.route('/code-file/')
def return_code_file():
    try:
        return send_file(os.path.join(APP_STATIC, 'codigo.txt'), attachment_filename='codigo.txt')
    except Exception as e:
        return str(e)

def calc_ajuste_rede(observation_matrix, std_patterns_matrix, ctrl_matrix, var_a_priori):
    # Variância a priori
    v_a_p = var_a_priori
    # Matriz de Pontos de Controle
    M_ctrl = ctrl_matrix

    # Cria vetor dos pontos de controle
    v_ctrl=[]
    for ctrl in M_ctrl:
        v_ctrl.append(ctrl[0])

    # Matriz de Observações
    M_obs = observation_matrix

    # Lista dos pontos que não são de controle
    v_nctrl=[]
    for observ in M_obs:
        if (observ[0] not in v_nctrl and observ[0] not in v_ctrl):
            v_nctrl.append(observ[0])
        if (observ[1] not in v_nctrl and observ[1] not in v_ctrl):
            v_nctrl.append(observ[1])

    # Matriz dos desvios padrões
    M_dev_pad = std_patterns_matrix

    def ajuste(A, P, L):
        N = np.dot(A.T, np.dot(P, A))
        U = np.dot(A.T, np.dot(P, L))
        Ninv = np.linalg.inv(N)
        delta_X = np.dot(Ninv, U)
        return delta_X

    #Aproximação inicial
    X0 = []
    for nctrl in v_nctrl:
        for obs in M_obs:
            if(obs[0] == nctrl and obs[1] in v_ctrl):
                for ctrl in M_ctrl:
                    if(ctrl[0]==obs[1]):
                        X0.append([obs[0],ctrl[1]+obs[2],ctrl[2]+obs[3],ctrl[3]+obs[4]])
                break
            elif (obs[1] == nctrl and obs[0] in v_ctrl):
                for ctrl in M_ctrl:
                    if(ctrl[0]==obs[0]):
                        X0.append([obs[1],ctrl[1]+obs[2],ctrl[2]+obs[3],ctrl[3]+obs[4]])
                break

    for ctrl in M_ctrl:
        X0.append([ctrl[0],ctrl[1], ctrl[2], ctrl[3]])

    #Matriz Jacobiana
    A_dev = []
    zero_line = []
    for x in v_nctrl:
        zero_line.append(0)
        zero_line.append(0)
        zero_line.append(0)
    for x in v_ctrl:
        zero_line.append(0)
        zero_line.append(0)
        zero_line.append(0)

    for obs in M_obs:
        A_dev.append(zero_line.copy())
        A_dev.append(zero_line.copy())
        A_dev.append(zero_line.copy())
        for i in range(len(X0)):
            if(obs[1] == X0[i][0]):
                A_dev[len(A_dev)-3][i*3] = 1
                A_dev[len(A_dev)-2][i*3+1] = 1
                A_dev[len(A_dev)-1][i*3+2] = 1
                break
        for i in range(len(X0)):
            if(obs[0] == X0[i][0]):
                A_dev[len(A_dev)-3][i*3] = -1
                A_dev[len(A_dev)-2][i*3+1] = -1
                A_dev[len(A_dev)-1][i*3+2] = -1
                break
    for ctrl in M_ctrl:
        A_dev.append(zero_line.copy())
        A_dev.append(zero_line.copy())
        A_dev.append(zero_line.copy())
        for i in range(len(X0)):
            if(ctrl[0] == X0[i][0]):
                A_dev[len(A_dev)-3][i*3] = 1
                A_dev[len(A_dev)-2][i*3+1] = 1
                A_dev[len(A_dev)-1][i*3+2] = 1
                break
    A_dev = np.array(A_dev)

    # Vetor de observações
    Lb = []
    for obs in M_obs:
        Lb.append([obs[2]])
        Lb.append([obs[3]])
        Lb.append([obs[4]])
    for ctrl in M_ctrl:
        Lb.append([ctrl[1]])
        Lb.append([ctrl[2]])
        Lb.append([ctrl[3]])

    # Vetor de observações estimadas com base na aproximação inicial (X0)
    L0 = []
    for obs in M_obs:
        valor = [0,0,0]
        for x in X0:
            if (x[0]==obs[1]):
                valor[0] += x[1]
                valor[1] += x[2]
                valor[2] += x[3]
            if (x[0]==obs[0]):
                valor[0] -= x[1]
                valor[1] -= x[2]
                valor[2] -= x[3]
        L0.append([valor[0]])
        L0.append([valor[1]])
        L0.append([valor[2]])
    for ctrl in M_ctrl:
        L0.append([ctrl[1]])
        L0.append([ctrl[2]])
        L0.append([ctrl[3]])

    # Vetor L
    #L = L0 - Lb
    L = np.array(L0) - np.array(Lb)

    # Matriz de Varianca e Covariancia
    diag_abs = np.array([])
    for dev_pad in M_dev_pad:
        diag_abs = np.concatenate((diag_abs, np.array([dev_pad[0]]), np.array([dev_pad[1]]),np.array([dev_pad[2]])))
    for ctrl in M_ctrl:
        diag_abs = np.concatenate((diag_abs, np.array([10**-12]), np.array([10**-12]),np.array([10**-12])))
    mvc_abs = np.diag(diag_abs ** 2)

    # Matriz Peso
    P_abs = np.linalg.inv(mvc_abs)

    # Aplicação do ajustamento
    delta_X = ajuste(A_dev, P_abs, L)
    #X = X0 - delta_X

    X1 = X0.copy()
    for i in range(len(X0)):
        X0[i]=[X0[i][1],X0[i][2],X0[i][3]]

    X0 = np.array(X0).reshape(3*len(X0),1)
    X = np.array(X0) - np.array(delta_X)

    # Matriz de variancia e covariancia
    N = np.dot(A_dev.T,np.dot(P_abs,A_dev))

    # Retorna matriz de parametros ajustados
    M_par_aj = []
    for i in range(len(X)):
        if i%3 == 0:
            if X1[i//3][0] not in v_ctrl:
                M_par_aj.append([X1[i//3][0], X[i][0], X[i+1][0], X[i+2][0]])

    # Retorna Matriz das Variancas e Covariancas dos parâmetros ajustados (para exibicao das elipses)
    M_var = (np.linalg.inv(N))
    sqrt_m_var = (np.linalg.inv(N))**0.5
    M_elipsoides_de_erro = []
    for i in range(len(M_var)):
        if i%3 == 0 and i < len(v_nctrl)*3:
            M_elipsoides_de_erro.append([X1[i//3][0], sqrt_m_var[i][0], sqrt_m_var[i+1][1], sqrt_m_var[i+2][2]])

    # Retorna a Matriz de Variância de Covariância
    MVC_final = []
    for i in range(len(M_var)):
        if(i < len(v_nctrl)*3):
            mvc_line = [v_nctrl[i//3]]
            for j in range(len(v_nctrl)):
                mvc_line.append(M_var[i][j])
                mvc_line.append(M_var[i][j+1])
                mvc_line.append(M_var[i][j+2])
            MVC_final.append(mvc_line.copy())

    # Realiza o Teste Qui-Quadrado
    V = np.dot(A_dev,X) - Lb

    quiquadrado_calc = np.array(np.dot(V.T,np.dot(P_abs,V))/v_a_p)[0][0]

    quitabmax = chi2.ppf(1-0.05/2, len(Lb) - len(X0))
    quitabmin = chi2.ppf(0.05/2, len(Lb) - len(X0))

    teste_hipotese_aceito = False
    if(quiquadrado_calc < quitabmax and quiquadrado_calc > quitabmin):
        teste_hipotese_aceito = True

    return M_par_aj, MVC_final, M_elipsoides_de_erro, teste_hipotese_aceito
