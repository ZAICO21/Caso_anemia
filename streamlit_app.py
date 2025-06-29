import streamlit as st
import pandas as pd
import networkx as nx
import folium
from streamlit_folium import folium_static
from geopy.distance import geodesic
from sklearn.neighbors import BallTree
import numpy as np

# Funciones

# Función para cargar los datos del CSV
@st.cache_data
def cargar_datos(ruta):
    df = pd.read_csv(ruta, delimiter=';')
    # Segmentación de datos con pandas
    df = df[(df['FECHA_REGISTRO'] > 20230720) & 
            (df['EDAD_REGISTRO'] >= 3) & 
            (df['EDAD_REGISTRO'] <= 10) & 
            (df['TIPO_EDAD'] == 'A')]
    return df.reset_index(drop=True)


# Función para calcular distancias entre viviendas con vecinos más cercanos
def calcular_distancias_viviendas(df, k_vecinos=25):
    coords = np.radians(df[['LATITUD', 'LONGITUD']].values)
    # Crea un BallTree con métrica haversine
    tree = BallTree(coords, metric='haversine')
    # Numero de vecinos a consultar
    k = k_vecinos + 1
    # Encuentra los k vecinos más cercanos y sus distancias
    distancias, indices = tree.query(coords, k=k)
    dist_km = distancias[:, 1:] * 6371
    indices_k = indices[:, 1:]

    data = []
    # Para cada punto, guartda los pares de distancia
    for i, (vecinos, dists) in enumerate(zip(indices_k, dist_km)):
        for j, dist in zip(vecinos, dists):
            id1 = df.iloc[i]['PK_REGISTRO']
            id2 = df.iloc[j]['PK_REGISTRO']
            data.append((id1, id2, round(dist, 3)))

    # Se crea el DataFrame con las distancias
    df_distancias = pd.DataFrame(data, columns=['PK_1', 'PK_2', 'DISTANCIA_KM'])
    df_distancias['ID'] = df_distancias.apply(lambda x: tuple(sorted([x['PK_1'], x['PK_2']])), axis=1)
    df_distancias = df_distancias.drop_duplicates('ID').drop(columns='ID').reset_index(drop=True)

    return df_distancias

# Función para conectar nodos de manera específica
def conectar_nodos(n1, n2, df, grafo):
        coord1 = tuple(df.iloc[n1][['LATITUD', 'LONGITUD']])
        coord2 = tuple(df.iloc[n2][['LATITUD', 'LONGITUD']])

        # Se calcula la distancia entre las coordenadas
        distancia_km = geodesic(coord1, coord2).km

        # Se agrega la arista al grafo con el peso de la distancia
        grafo.add_edge(n1, n2, weight=round(distancia_km, 3))

# Función para construir el grafo a partir de los dataframes
def construir_grafo(df_anemia, df_distancia):
    G = nx.Graph()
    colores = []
    pk_to_index = df_anemia.reset_index()[['index', 'PK_REGISTRO']].set_index('PK_REGISTRO')['index'].to_dict()

    # Agregar nodos con coordenadas, el código de registro y creamos el arreglo de colores
    for idx, row in df_anemia.iterrows():
        G.add_node(idx, pos_lat=row['LATITUD'], pos_lon=row['LONGITUD'], cod=row['PK_REGISTRO'])
        if row['GRADO_SEVERIDAD'] == 'LEV':
            colores.append('green')
        elif row['GRADO_SEVERIDAD'] == 'MOD':
            colores.append('orange')
        elif row['GRADO_SEVERIDAD'] == 'SEV':
            colores.append('red')

    # Conectar nodos según las distancias
    for _, row in df_distancia.iterrows():
        pk1 = row['PK_1']
        pk2 = row['PK_2']
        dist = row['DISTANCIA_KM']

        idx1 = pk_to_index.get(pk1)
        idx2 = pk_to_index.get(pk2)

        if idx1 is not None and idx2 is not None:
            G.add_edge(idx1, idx2, weight=dist)    

    # Conectar nodos específicos
    conectar_nodos(1494, 447, df_anemia, G)
    conectar_nodos(553, 1494, df_anemia, G)

        
    return G, colores

# Función que genera el mapa inicial
def generar_mapa(G, colores, df):
    # Se inicializa el mapa
    mapa = folium.Map(location=[-6.5, -76.5], zoom_start=8)

    # Se recorre el grafo para mostrar los nodos en el mapa
    for n, d in G.nodes(data=True):
        
        # Se usará el marcador de círculo para representar los nodos
        folium.CircleMarker(
            location=(d['pos_lat'], d['pos_lon']),
            radius=4,
            color=colores[n],
            fill=True,
            fill_color=colores[n],
            fill_opacity=0.9,
            tooltip=f"Nodo: {n}<br>Severidad: {df.iloc[n]['GRADO_SEVERIDAD']}<br>Hospital: {df.iloc[n]['NOMBRE_ESTABLECIMIENTO']}"
        ).add_to(mapa)

    return mapa

def generar_mapa_ruta(G, colores, nodos_ruta, df):

    # Verifica si la ruta está vacía
    if not nodos_ruta:
        return folium.Map(location=[-6.5, -76.5], zoom_start=8)

    # Inicializa el mapa centrado en el primer nodo
    lat, lon = G.nodes[nodos_ruta[0]]['pos_lat'], G.nodes[nodos_ruta[0]]['pos_lon']
    mapa = folium.Map(location=[lat, lon], zoom_start=8)

    # Agregar los nodos del camino/ruta, se usará un marcador para su representación
    for n in nodos_ruta:
        data = G.nodes[n]
        folium.Marker(
            location=(data['pos_lat'], data['pos_lon']),
            icon=folium.Icon(color=colores[n]),
            tooltip=f"Nodo: {n}<br>Severidad: {df.iloc[n]['GRADO_SEVERIDAD']}<br>Hospital: {df.iloc[n]['NOMBRE_ESTABLECIMIENTO']}"
        ).add_to(mapa)

    # Se dibujan las aristas de la ruta
    for i in range(len(nodos_ruta)-1):
        n1, n2 = nodos_ruta[i], nodos_ruta[i+1]
        loc1 = (G.nodes[n1]['pos_lat'], G.nodes[n1]['pos_lon'])
        loc2 = (G.nodes[n2]['pos_lat'], G.nodes[n2]['pos_lon'])
        folium.PolyLine([loc1, loc2], color='blue', weight=3).add_to(mapa)

    return mapa

# Página de Streamlit

# Inicialización de variables
ruta = "ANEMIA_DA.csv"
Data_Anemia = cargar_datos(ruta)
df_distancias = calcular_distancias_viviendas(Data_Anemia)
G,color = construir_grafo(Data_Anemia,df_distancias)
mapa = generar_mapa(G, color, Data_Anemia)

# Interfaz de usuario
st.title("Anemia en el departamento de San Martín")

st.subheader("""Integrantes del equipo: 
            \n  - Gianmarco Fabian Jiménez Guerra - U202123843
            \n  - Alexander Felipe Vasquez Roncal - U202222473
            \n  - Eric Marlon Olivera Barzola - U202315032""")

st.subheader("📍 Mapa de conexiones geográficas de pacientes")
folium_static(mapa)

# Selección de opciones de algoritmos
algoritmo = st.selectbox("Selecciona el algoritmo: ", ['Dijkstra', 'BFS', 'Kruskal'])

if algoritmo == 'Dijkstra':
    # Nodo origen y destino
    origen = st.number_input("Nodo origen", min_value=0, max_value=len(G.nodes)-1, step=1)
    destino = st.number_input("Nodo destino", min_value=0, max_value=len(G.nodes)-1, step=1)

    if st.button("Calcular ruta más corta"):
        try:
            # Calcular la ruta más corta usando Dijkstra
            ruta_dijkstra = nx.dijkstra_path(G, source=origen, target=destino)
            mapa_ruta = generar_mapa_ruta(G, color, ruta_dijkstra, Data_Anemia)
            folium_static(mapa_ruta)
        except Exception as e:
            # Manejo de error
            st.error(f"No es posible calcular la ruta más corta: {e}")

elif algoritmo == 'BFS':
    # Nodo origen y pasos
    origen = st.number_input("Nodo", min_value=0, max_value=len(G.nodes)-1, step=1)
    max_pasos = st.number_input("Pasos", min_value=1, max_value=10, step=1)

    if st.button("Calcular BFS"):
        try:
            # Calcular BFS desde el nodo de origen
            bfs_edges = list(nx.bfs_edges(G, source=origen, depth_limit=max_pasos))
            # Obtener los nodos en orden del recorrido
            nodos_bfs = [origen] + [v for u, v in bfs_edges]
            mapa_ruta = generar_mapa_ruta(G, color, nodos_bfs, Data_Anemia)
            folium_static(mapa_ruta)
        except Exception as e:
            # Manejo de error
            st.error(f"Error al calcular BFS: {e}")

elif algoritmo == 'Kruskal':
    if st.button("Calcular MST"):
        try:
            # Filtrar los nodos de casos severos
            nodos_severos = [n for n, d in G.nodes(data=True) if Data_Anemia.iloc[n]['GRADO_SEVERIDAD'] == 'SEV']
            # Se crea un grafo con los nodos severos
            grafo_severos = nx.Graph()
            for n in nodos_severos:
                grafo_severos.add_node(
                    n,
                    pos_lat=G.nodes[n]['pos_lat'],
                    pos_lon=G.nodes[n]['pos_lon']
                )

            # Se conectan todos los nodos entre sí
            for i in range(len(nodos_severos)):
                for j in range(i+1, len(nodos_severos)):
                    n1, n2 = nodos_severos[i], nodos_severos[j]
                    coord1 = tuple(Data_Anemia.iloc[n1][['LATITUD', 'LONGITUD']])
                    coord2 = tuple(Data_Anemia.iloc[n2][['LATITUD', 'LONGITUD']])
                    distancia_km = geodesic(coord1, coord2).km
                    grafo_severos.add_edge(n1, n2, weight=round(distancia_km, 3))

            # Se calcula el MST usando Kruskal
            mst = nx.minimum_spanning_tree(grafo_severos)
            nodos_mst = list(mst.nodes)
            edges_mst = list(mst.edges)

            if nodos_mst:
                lat0 = grafo_severos.nodes[nodos_mst[0]]['pos_lat']
                lon0 = grafo_severos.nodes[nodos_mst[0]]['pos_lon']
                mapa_mst = folium.Map(location=[lat0, lon0], zoom_start=8)

                # Se muestra cada nodo del MST
                for n in nodos_mst:
                    data = grafo_severos.nodes[n]
                    folium.Marker(
                        location=(data['pos_lat'], data['pos_lon']),
                        icon=folium.Icon(color=color[n]),
                        tooltip=f"Nodo: {n}<br>Severidad: {Data_Anemia.iloc[n]['GRADO_SEVERIDAD']}"
                    ).add_to(mapa_mst)

                # Se dibujan las aristas del MST
                for n1, n2 in edges_mst:
                    loc1 = (grafo_severos.nodes[n1]['pos_lat'], grafo_severos.nodes[n1]['pos_lon'])
                    loc2 = (grafo_severos.nodes[n2]['pos_lat'], grafo_severos.nodes[n2]['pos_lon'])
                    folium.PolyLine([loc1, loc2], color='blue', weight=3).add_to(mapa_mst)

                folium_static(mapa_mst)
            else:
                st.warning("El MST no contiene nodos.")
        except Exception as e:
            st.warning(f"Error al calcular el MST: {e}")


