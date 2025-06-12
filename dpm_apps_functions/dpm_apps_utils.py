# __________________________________________________________________________________________________________________________________________________________
# files_path_collect_df
# __________________________________________________________________________________________________________________________________________________________
def files_path_collect_df(config: dict) -> "pd.DataFrame":
    """
    Busca archivos de video en una ruta (local o Google Drive), extrae sus propiedades usando ffprobe
    y devuelve un DataFrame con los resultados. 

    Args:
        config (dict):
            - video_files_root_path (str): Ruta raíz donde buscar archivos de video.
            - video_files_target_search_folder (list): Lista de subcarpetas de interés.
            - video_files_target_search_extension (list): Lista de extensiones de archivo (e.g., [".mp4"]).
            - ini_environment_identificated (str): Entorno de ejecución.
            - json_keyfile_local (str): Ruta del JSON de credenciales LOCAL.
            - json_keyfile_colab (str): Ruta del JSON de credenciales COLAB.
            - json_keyfile_GCP_secret_id (str): ID de secreto GCP.

    Returns:
        pd.DataFrame: DataFrame con la información y metadatos de los videos.

    Raises:
        ValueError: Si falta algún parámetro obligatorio o no se encuentran archivos.
        NotImplementedError: Si la ruta es de Google Drive.
        Exception: Para errores inesperados.
    """
    import os
    import subprocess
    import json
    import pandas as pd
    from datetime import datetime
    from time import time
    from typing import Optional
    try:
        from IPython.display import clear_output
    except ImportError:
        def clear_output(wait=False):
            os.system('cls' if os.name == 'nt' else 'clear')

    # Contador de líneas impresas
    line_counter = 0
    max_lines = 200
    def safe_print(*args, **kwargs):
        nonlocal line_counter
        if line_counter >= max_lines:
            clear_output(wait=True)
            line_counter = 0
        print(*args, **kwargs)
        line_counter += 1

    try:
        safe_print("\n[PROCESS START ▶️] Iniciando recolección de archivos.", flush=True)
        start_time = time()

        # Validación parámetros
        video_root_path: Optional[str] = config.get("video_files_root_path")
        if not video_root_path:
            raise ValueError("Falta 'video_files_root_path' en config.")
        video_folders = config.get("video_files_target_search_folder", [])
        video_exts = config.get("video_files_target_search_extension", [])
        if not video_exts:
            raise ValueError("Falta 'video_files_target_search_extension' o está vacío.")
        env_ident: Optional[str] = config.get("ini_environment_identificated")
        if not env_ident:
            raise ValueError("Falta 'ini_environment_identificated' en config.")
        if env_ident == "LOCAL":
            keyfile = config.get("json_keyfile_local")
        elif env_ident == "COLAB":
            keyfile = config.get("json_keyfile_colab")
        else:
            keyfile = config.get("json_keyfile_GCP_secret_id")
        if not keyfile:
            raise ValueError("Falta clave de credenciales para el entorno.")
        safe_print("[VALIDATION SUCCESS ✅] Parámetros validados.", flush=True)

        # Ruta Google Drive no soportada
        if video_root_path.startswith("https://"):
            safe_print("[WARNING ⚠️] Google Drive no implementado.", flush=True)
            raise NotImplementedError("Extracción desde Google Drive no está implementada.")

        # Búsqueda de archivos
        def _find_files(root, folders, exts):
            results = []
            found = 0
            for cur_root, dirs, files in os.walk(root):
                base = os.path.basename(cur_root)
                if folders and base not in folders:
                    continue
                for f in files:
                    _, ext = os.path.splitext(f)
                    if ext.lower() in (e.lower() for e in exts):
                        path = os.path.join(cur_root, f)
                        results.append({"video_file_path": path, "video_file_name": f})
                        found += 1
                        safe_print(f"[FOUND] {found}: {f}", flush=True)
            if not results:
                return pd.DataFrame()
            return pd.DataFrame(results)

        safe_print(f"[SEARCH ▶️] Ruta: {video_root_path} | Exts: {video_exts}", flush=True)
        df_paths = _find_files(video_root_path, video_folders, video_exts)
        if df_paths.empty:
            raise ValueError("No se encontraron archivos.")
        safe_print(f"[FOUND ✅] {len(df_paths)} archivos.", flush=True)

        # Extracción metadatos
        total = len(df_paths)
        rows = []
        safe_print("[EXTRACT ▶️] Extrayendo metadatos.", flush=True)
        for idx, path in enumerate(df_paths['video_file_path'], 1):
            meta = {}
            try:
                size_mb = os.path.getsize(path) // (1024*1024)
                proc = subprocess.run(
                    ['ffprobe','-v','error','-print_format','json','-show_streams','-show_format', path],
                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=True)
                info = json.loads(proc.stdout)
                # valores por defecto
                meta = {
                    'file_name': os.path.basename(path),
                    'file_path': path,
                    'file_creation_date': datetime.fromtimestamp(os.path.getctime(path)),
                    'file_last_modified_date': datetime.fromtimestamp(os.path.getmtime(path)),
                    'file_scrap_date': datetime.now(),
                    'file_size_mb': size_mb,
                    'duration_hms':'00:00:00','duration_ms':0,
                    'video_codec':None,'video_bitrate_kbps':0,'video_fps':0,'video_resolution':'unknown',
                    'audio_codec':None,'audio_bitrate_kbps':0,'audio_channels':0,'audio_sample_rate_hz':0
                }
                for st in info.get('streams',[]):
                    if st.get('codec_type')=='video':
                        meta['video_codec']=st.get('codec_name')
                        meta['video_bitrate_kbps']=int(st.get('bit_rate',0))//1000
                        w,h=st.get('width'),st.get('height')
                        meta['video_resolution']=f"{w}x{h}" if w and h else 'unknown'
                        n,d=map(int,st.get('r_frame_rate','0/1').split('/'))
                        meta['video_fps']=n/d if d else 0
                    elif st.get('codec_type')=='audio':
                        meta['audio_codec']=st.get('codec_name')
                        meta['audio_bitrate_kbps']=int(st.get('bit_rate',0))//1000
                        meta['audio_channels']=st.get('channels',0)
                        meta['audio_sample_rate_hz']=int(st.get('sample_rate',0))
                fmt = info.get('format',{})
                dur = float(fmt.get('duration', 0))
                meta['duration_ms']=int(dur*1000)
                h=int(dur)//3600; m=(int(dur)%3600)//60; s=int(dur)%60
                meta['duration_hms']=f"{h:02d}:{m:02d}:{s:02d}"
                safe_print(f"[META] {meta['file_name']} | {meta['duration_hms']} | {meta['video_resolution']}", flush=True)
            except Exception as e:
                safe_print(f"[ERROR] {os.path.basename(path)}: {e}", flush=True)
            rows.append(meta)
            # progreso
            pct = idx/total*100
            bar = '['+'='*int(pct//5)+' '*(20-int(pct//5))+']'
            safe_print(f"[PROG] {bar} {pct:.1f}% ({idx}/{total})", flush=True)

        safe_print("[EXTRACT ✅] Metadatos extraídos.", flush=True)
        df = pd.DataFrame(rows)[[
            'file_name','file_path','file_creation_date','file_last_modified_date','file_scrap_date',
            'file_size_mb','duration_hms','duration_ms','video_codec','video_bitrate_kbps',
            'video_fps','video_resolution','audio_codec','audio_bitrate_kbps',
            'audio_channels','audio_sample_rate_hz'
        ]]
        for c in ['file_creation_date','file_last_modified_date','file_scrap_date']:
            df[c]=pd.to_datetime(df[c], errors='coerce')
        df.to_csv('video_files_backup.csv', index=False)
        safe_print(f"[BACKUP] Guardado en video_files_backup.csv", flush=True)
        elapsed = time() - start_time
        safe_print(f"[END ✅] {len(df)} videos en {elapsed:.2f}s", flush=True)
        return df
    except ValueError as ve:
        safe_print(f"[FAIL] {ve}", flush=True)
        return None
    except Exception as e:
        safe_print(f"[FAIL] Inesperado: {e}", flush=True)
        return None


    





















# __________________________________________________________________________________________________________________________________________________________
# df_to_whisper_transcribe_to_spreadsheet
# __________________________________________________________________________________________________________________________________________________________
def df_to_whisper_transcribe_to_spreadsheet(config: dict) -> None:
    """
    Transcribe archivos de vídeo usando un modelo Whisper y escribe los resultados en una hoja de cálculo de Google Sheets.

    Args:
        config (dict): Diccionario de configuración que debe incluir:
            - source_files_path_table_df (pd.DataFrame): DataFrame con al menos la columna especificada en 'field_name_for_file_path'.
            - target_files_path_table_spreadsheet_url (str): URL de la hoja de cálculo destino.
            - target_files_path_table_spreadsheet_worksheet (str): Nombre de la worksheet destino.
            - field_name_for_file_path (str): Nombre de la columna con la ruta del vídeo.
            - whisper_model_size (str): Tamaño del modelo Whisper (ej. "small", "medium", etc.).
            - whisper_language (str, opcional): Idioma de la transcripción (default: "en").
            - ini_environment_identificated (str): Identificador del entorno ("LOCAL", "COLAB", "COLAB_ENTERPRISE" o un project_id).
            - json_keyfile_local (str): Ruta del archivo JSON de credenciales para entorno LOCAL.
            - json_keyfile_colab (str): Ruta del archivo JSON de credenciales para entorno COLAB.
            - json_keyfile_GCP_secret_id (str): ID del secreto para entornos GCP.

    Returns:
        None

    Raises:
        ValueError: Si falta algún parámetro obligatorio o si el DataFrame fuente está vacío o no contiene la columna requerida.
        Exception: Para otros errores inesperados.
    """
    import os
    import time
    import gspread
    from oauth2client.service_account import ServiceAccountCredentials
    from datetime import datetime
    import whisper
    import torch

    def _trocear_texto(texto: str, max_chars: int = 50000, max_partes: int = 10) -> list:
        """
        Trocea un texto en partes de longitud <= max_chars y retorna una lista de longitud max_partes.
        """
        trozos = [texto[i:i + max_chars] for i in range(0, len(texto), max_chars)]
        trozos = trozos[:max_partes]
        if len(trozos) < max_partes:
            trozos += [""] * (max_partes - len(trozos))
        return trozos

    def _process_transcription() -> None:
        # Validación inicial de parámetros
        required_keys = [
            "source_files_path_table_df",
            "target_files_path_table_spreadsheet_url",
            "target_files_path_table_spreadsheet_worksheet",
            "field_name_for_file_path",
            "whisper_model_size"
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"[VALIDATION [ERROR ❌]] Falta el parámetro obligatorio: '{key}'.")
            # Para el DataFrame se evita evaluar su veracidad de forma ambigua
            if key != "source_files_path_table_df" and not config.get(key):
                raise ValueError(f"[VALIDATION [ERROR ❌]] El parámetro '{key}' está vacío.")

        # Extraer parámetros
        source_df = config["source_files_path_table_df"]
        if source_df is None:
            raise ValueError("[VALIDATION [ERROR ❌]] El DataFrame fuente no puede ser None.")
        if source_df.empty:
            raise ValueError("[VALIDATION [ERROR ❌]] El DataFrame fuente está vacío.")
        field_name = config["field_name_for_file_path"]
        if field_name not in source_df.columns:
            raise ValueError(f"[VALIDATION [ERROR ❌]] La columna '{field_name}' no existe en el DataFrame fuente.")

        target_spreadsheet_url = config["target_files_path_table_spreadsheet_url"]
        target_worksheet_name = config["target_files_path_table_spreadsheet_worksheet"]
        whisper_model_size = config["whisper_model_size"]
        whisper_language = config.get("whisper_language", "en")

        # Configurar credenciales según el entorno
        ini_env = config.get("ini_environment_identificated")
        if ini_env == "LOCAL":
            credentials_path = config.get("json_keyfile_local")
        elif ini_env == "COLAB":
            credentials_path = config.get("json_keyfile_colab")
        else:
            credentials_path = config.get("json_keyfile_GCP_secret_id")
        if not credentials_path:
            raise ValueError("[VALIDATION [ERROR ❌]] Credenciales no definidas para el entorno especificado.")

        # Autenticación con Google Sheets
        print("🔹🔹🔹 [START ▶️] Autenticando con Google Sheets 🔹🔹🔹", flush=True)
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
        client = gspread.authorize(creds)
        print("[AUTH SUCCESS ✅] Autenticación exitosa.", flush=True)

        # Preparar la hoja destino
        print(f"🔹🔹🔹 [START ▶️] Preparando hoja destino: {target_worksheet_name} 🔹🔹🔹", flush=True)
        spreadsheet_dest = client.open_by_url(target_spreadsheet_url)
        try:
            destination_sheet = spreadsheet_dest.worksheet(target_worksheet_name)
        except gspread.WorksheetNotFound:
            print("[INFO ℹ️] Hoja destino no existe. Creándola...", flush=True)
            destination_sheet = spreadsheet_dest.add_worksheet(title=target_worksheet_name, rows=1000, cols=30)
        destination_sheet.clear()
        dest_header = (
            ["file_path", "transcription_date", "transcription_duration", "whisper_model", "GPU_model"] +
            [f"transcription_part_{i}" for i in range(1, 11)] +
            [f"transcription_seg_part_{i}" for i in range(1, 11)]
        )
        destination_sheet.update("A1", [dest_header])
        print("[SHEET SUCCESS ✅] Hoja destino preparada y encabezados definidos.", flush=True)

        # Cargar modelo Whisper
        print(f"🔹🔹🔹 [START ▶️] Cargando modelo Whisper '{whisper_model_size}' 🔹🔹🔹", flush=True)
        model = whisper.load_model(whisper_model_size)
        gpu_model = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No"
        print("[MODEL SUCCESS ✅] Modelo Whisper cargado.", flush=True)

        # Convertir DataFrame a lista de diccionarios
        source_data = source_df.to_dict(orient="records")
        total_rows = len(source_data)
        print(f"[DATA INFO ℹ️] DataFrame fuente cargado. Total filas: {total_rows}", flush=True)

        # Procesar cada fila
        for idx, row_data in enumerate(source_data, start=1):
            video_path_value = row_data.get(field_name, "")
            if not video_path_value:
                continue

            print(f"[PROCESSING 🔄] ({idx}/{total_rows}) Transcribiendo: {video_path_value} (idioma='{whisper_language}')", flush=True)
            start_time_proc = time.time()
            try:
                result = model.transcribe(video_path_value, language=whisper_language)
                transcription_full = result.get("text", "")
                transcription_segments_full = "".join(
                    [f"[{seg['start']:.2f}s - {seg['end']:.2f}s]: {seg['text']}\n" for seg in result.get("segments", [])]
                )
            except Exception as e:
                print(f"[ERROR ❌] Error al transcribir {video_path_value}: {e}", flush=True)
                continue

            duration = round(time.time() - start_time_proc, 2)
            transcription_parts = _trocear_texto(transcription_full)
            transcription_seg_parts = _trocear_texto(transcription_segments_full)
            transcription_date = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            row_to_insert = (
                [video_path_value, transcription_date, duration, whisper_model_size, gpu_model] +
                transcription_parts + transcription_seg_parts
            )

            try:
                destination_sheet.append_row(row_to_insert, value_input_option="USER_ENTERED")
                print(f"[WRITE SUCCESS ✅] Fila {idx} escrita correctamente.", flush=True)
            except Exception as e:
                print(f"[ERROR ❌] Error al escribir la fila {idx}: {e}", flush=True)
                continue

        print("🔹🔹🔹 [FINISHED ✅] Proceso de transcripción completado.", flush=True)

    # Ejecutar el proceso de transcripción con manejo de errores
    try:
        _process_transcription()
        print("Proceso completado con éxito.", flush=True)
    except ValueError as ve:
        print(f"Error en los parámetros: {ve}", flush=True)
    except Exception as e:
        print(f"Error inesperado: {e}", flush=True)

















# __________________________________________________________________________________________________________________________________________________________
# LLM_process_text
# __________________________________________________________________________________________________________________________________________________________
def LLM_process_text(params: dict) -> None:
    """
    Procesa filas de un DataFrame utilizando un modelo LLM según el prompt proporcionado,
    y escribe progresivamente los resultados en una hoja de cálculo de Google Sheets.

    Además, se capturan los tokens consumidos (prompt_tokens, completion_tokens, total_tokens)
    por cada fila, y se registran en logs detallados para facilitar la depuración.

    Parámetros esperados en `params`:
      - "source_table_df": DataFrame con los textos a procesar.
      - "source_table_field_name": Nombre de la columna con el texto.
      - "system_prompt": URL desde donde se descarga el prompt del sistema para la primera petición.
      - "LLM_API_key_GCP_secret_manager_name": Clave de API para el modelo LLM. Si se proporciona la clave "LLM_API_key_GCP_secret_manager_project_id",
                         se obtiene el valor de Secret Manager usando el secreto "OpenAI_API_key".
      - "LLM_API_key_GCP_secret_manager_project_id" (opcional): project_id para acceder al secreto en GCP.
      - "target_table_spreadsheet_url": URL de la hoja de cálculo destino.
      - "target_table_spreadsheet_worksheet": Nombre de la hoja de cálculo destino.
      - "target_table_field_LLM_response_name": Nombre del campo para la respuesta limpia.
      - "target_table_field_LLM_comments_name": Nombre del campo para comentarios.
      - "target_table_field_LLM_response_comments_sep": Separador para dividir respuesta y comentarios.
      - "target_table_filed_to_keep_list": Lista de campos a conservar y su orden.
      - "ConversationBufferMemory_params": Parámetros para la memoria conversacional.
      - (Opcional) "system_prompt_second_and_later": Texto para el system prompt a partir de la segunda fila.
    """
    import time
    import pandas as pd
    import requests
    from datetime import datetime

    from langchain_openai import ChatOpenAI
    from langchain.prompts import (
        SystemMessagePromptTemplate,
        HumanMessagePromptTemplate,
        ChatPromptTemplate
    )
    from langchain.chains import LLMChain
    from langchain.memory import ConversationBufferWindowMemory
    from langchain.schema import LLMResult, Generation

    import gspread
    from oauth2client.service_account import ServiceAccountCredentials

    # Función auxiliar para contar tokens (aproximación: usando número de palabras)
    def count_tokens(text: str) -> int:
        return len(text.split())

    # VALIDACIÓN DE PARÁMETROS
    def _validate_params() -> None:
        required_params = [
            "source_table_df",
            "source_table_field_name",
            "system_prompt",
            "target_table_spreadsheet_url",
            "target_table_spreadsheet_worksheet",
            "target_table_field_LLM_response_name",
            "target_table_field_LLM_comments_name",
            "target_table_field_LLM_response_comments_sep",
            "target_table_filed_to_keep_list"
        ]
        for req in required_params:
            if req not in params:
                raise ValueError(f"[VALIDATION [ERROR ❌]] Falta el parámetro esencial '{req}' en 'params'.")
            if req == "source_table_df":
                df = params["source_table_df"]
                if not isinstance(df, pd.DataFrame) or df.empty:
                    raise ValueError("[VALIDATION [ERROR ❌]] El DataFrame de entrada está vacío o no es válido.")
            else:
                if not params[req]:
                    raise ValueError(f"[VALIDATION [ERROR ❌]] El parámetro '{req}' está vacío o no es válido.")

    # RECUPERAR LLM_API_key_GCP_secret_manager_name DESDE SECRET MANAGER (si se define LLM_API_key_GCP_secret_manager_project_id)
    def _retrieve_llm_api_key_from_secret_manager() -> None:
        # Se requiere obligatoriamente el project_id para acceder al secreto
        project_id = params.get("LLM_API_key_GCP_secret_manager_project_id")
        if not project_id:
            raise ValueError("[VALIDATION [ERROR ❌]] Falta 'LLM_API_key_GCP_secret_manager_project_id' en params. Este parámetro es obligatorio para acceder al Secret Manager.")
        
        import os
        from google.cloud import secretmanager

        ini_env = params.get("ini_environment_identificated")
        if ini_env == "LOCAL":
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = params.get("json_keyfile_local")
        elif ini_env == "COLAB":
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = params.get("json_keyfile_colab")
        else:
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = params.get("json_keyfile_GCP_secret_id")
            
        secret_full_name = f"projects/{project_id}/secrets/OpenAI_API_key/versions/latest"
        print(f"[SECRET MANAGER INFO ℹ️] Accediendo al secreto: {secret_full_name}", flush=True)
        try:
            client = secretmanager.SecretManagerServiceClient()
            response = client.access_secret_version(request={"name": secret_full_name})
            params["LLM_API_key_GCP_secret_manager_name"] = response.payload.data.decode("UTF-8")
            print("[SECRET MANAGER SUCCESS ✅] LLM_API_key obtenida correctamente desde Secret Manager.", flush=True)
        except Exception as e:
            raise ValueError(f"[SECRET MANAGER ERROR ❌] Error al obtener LLM_API_key: {e}")


    # AUTENTICACIÓN CON GOOGLE SHEETS
    def _auth_with_google_sheets() -> gspread.Worksheet:
        ini_env = params.get("ini_environment_identificated")
        if ini_env == "LOCAL":
            credentials_path = params.get("json_keyfile_local")
            print("[AUTH INFO ℹ️] Entorno LOCAL: usando json_keyfile_local.", flush=True)
        elif ini_env == "COLAB":
            credentials_path = params.get("json_keyfile_colab")
            print("[AUTH INFO ℹ️] Entorno COLAB: usando json_keyfile_colab.", flush=True)
        elif ini_env == "COLAB_ENTERPRISE":
            credentials_path = params.get("json_keyfile_GCP_secret_id")
            print("[AUTH INFO ℹ️] Entorno COLAB_ENTERPRISE: usando json_keyfile_GCP_secret_id.", flush=True)
        else:
            credentials_path = params.get("json_keyfile_GCP_secret_id")
            print("[AUTH WARNING ⚠️] Entorno no reconocido. Se asume GCP secret ID.", flush=True)
        if not credentials_path:
            print("[AUTH WARNING ⚠️] No se ha definido ruta o ID de credenciales. Se intentará sin credenciales.", flush=True)
        scope = ["https://www.googleapis.com/auth/spreadsheets", "https://www.googleapis.com/auth/drive"]
        creds = ServiceAccountCredentials.from_json_keyfile_name(credentials_path, scope)
        client = gspread.authorize(creds)
        spreadsheet_dest = client.open_by_url(params["target_table_spreadsheet_url"])
        worksheet_name = params["target_table_spreadsheet_worksheet"]
        try:
            sheet = spreadsheet_dest.worksheet(worksheet_name)
        except gspread.WorksheetNotFound:
            print("[SHEET INFO ℹ️] Worksheet no encontrado, creando uno nuevo...", flush=True)
            sheet = spreadsheet_dest.add_worksheet(title=worksheet_name, rows="1000", cols="30")
        return sheet

    # ESCRIBIR ENCABEZADO EN LA HOJA DE CÁLCULO (corregido para evitar DeprecationWarning)
    def _append_sheet_header(sheet: gspread.Worksheet) -> None:
        header = params["target_table_filed_to_keep_list"]
        sheet.clear()
        # Usar argumentos nombrados para evitar el DeprecationWarning
        sheet.update(values=[header], range_name="A1")
        print("[SHEET INFO ℹ️] Worksheet limpia y encabezado escrito.", flush=True)

    # DESCARGAR Y PREPARAR SYSTEM PROMPT
    def _get_system_prompt_text() -> str:
        system_prompt_input = params["system_prompt"]
        if system_prompt_input.startswith("http") and "github.com" in system_prompt_input:
            system_prompt_input = system_prompt_input.replace("github.com", "raw.githubusercontent.com").replace("/blob/", "/")
        try:
            response = requests.get(system_prompt_input)
            response.raise_for_status()
        except Exception as e:
            raise ValueError(f"[VALIDATION [ERROR ❌]] No se pudo obtener el system_prompt desde {system_prompt_input}: {e}")
        return response.text.replace("{className}", "").replace("{message}", "")

    # CONFIGURAR LLM Y MEMORIA (configuración modular del prompt)
    def _configure_llm_chain(system_prompt_text: str) -> LLMChain:
        model_name = params.get("LLM_model_name", params.get("model_name"))
        temperature = params.get("LLM_temperature", params.get("temperature"))
        api_key = params["LLM_API_key_GCP_secret_manager_name"]

        mem_params = params.get("ConversationBufferMemory_params", {})
        memory = ConversationBufferWindowMemory(**mem_params)
        if mem_params.get("memory_flush_at_start", False):
            print("[MEMORY INFO ℹ️] Reseteando memoria conversacional.", flush=True)
            memory.clear()

        print(f"[CHAIN CONFIG INFO ℹ️] Configurando LLMChain con modelo '{model_name}' y temperatura {temperature}.", flush=True)
        llm = ChatOpenAI(api_key=api_key, model_name=model_name, temperature=temperature)
        sys_template = SystemMessagePromptTemplate.from_template(system_prompt_text, template_format="jinja2")
        human_template = HumanMessagePromptTemplate.from_template("Aquí tienes el contenido (texto completo):\n{content}")
        chat_prompt = ChatPromptTemplate.from_messages([sys_template, human_template])
        chat_prompt.input_variables = ["content"]
        return LLMChain(llm=llm, prompt=chat_prompt, memory=memory)

    # PROCESAR UNA FILA Y ESCRIBIR RESULTADO EN GOOGLE SHEETS (con desglose de tokens)
    def _process_row_and_write(chain: LLMChain, sheet: gspread.Worksheet, row_data: dict, row_index: int) -> tuple:
        field = params["source_table_field_name"]
        content = row_data.get(field, "").strip()
        if not content:
            print(f"\n🔹🔹🔹 [SKIP ▶️] Fila {row_index} sin contenido. Se omite. 🔹🔹🔹", flush=True)
            return (False, 0.0, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        print(f"\n🔹🔹🔹 [START ▶️] Procesando fila {row_index}. Longitud contenido: {len(content)} 🔹🔹🔹", flush=True)
        t_start = time.time()
        result = chain.generate([{"content": content}])
        duration = round(time.time() - t_start, 2)
        generation = result.generations[0][0].text.strip() if result.generations else ""
        usage_info = result.llm_output.get("token_usage", {}) if result.llm_output else {}
        prompt_tokens = usage_info.get("prompt_tokens", 0)
        completion_tokens = usage_info.get("completion_tokens", 0)
        total_tokens = usage_info.get("total_tokens", 0)

        print(f"[LLM RESPONSE SUCCESS ✅] Fila {row_index} procesada en {duration} s.", flush=True)
        print(f"[TOKENS USAGE ℹ️] prompt_tokens={prompt_tokens}, completion_tokens={completion_tokens}, total_tokens={total_tokens}", flush=True)

        sep = params["target_table_field_LLM_response_comments_sep"]
        parts = generation.split(sep, 1)
        transcription_clean = parts[0].strip()
        comments = parts[1].strip() if len(parts) > 1 else ""

        def _trocear_texto(texto: str, max_chars: int = 50000, max_parts: int = 10) -> list:
            pieces = [texto[i:i + max_chars] for i in range(0, len(texto), max_chars)]
            pieces = pieces[:max_parts]
            if len(pieces) < max_parts:
                pieces += [""] * (max_parts - len(pieces))
            return pieces

        transcription_parts = _trocear_texto(transcription_clean)
        current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        target_fields = params["target_table_filed_to_keep_list"]
        row_final = []
        for key in target_fields:
            if key in row_data:
                row_final.append(row_data.get(key, ""))
            elif key == "transcription_clean_date":
                row_final.append(current_timestamp)
            elif key == "transcription_clean_duration":
                row_final.append(duration)
            elif key == "transcription_clean_comments":
                row_final.append(transcription_clean)
            elif key.startswith("transciption_clean_part_"):
                try:
                    part_num = int(key.split("_")[-1])
                    row_final.append(transcription_parts[part_num - 1])
                except Exception:
                    row_final.append("")
            elif key == "comments":
                row_final.append(comments)
            elif key == "LLM_model_name":
                row_final.append(params.get("LLM_model_name", ""))
            elif key == "LLM_temperature":
                row_final.append(params.get("LLM_temperature", ""))
            else:
                row_final.append("")

        try:
            sheet.append_row(row_final, value_input_option="USER_ENTERED")
            print(f" [SUCCESS ✅] Fila {row_index} escrita en Google Sheets.", flush=True)
        except Exception as e:
            print(f" [ERROR ❌] Error al escribir la fila {row_index}: {e}", flush=True)
        return (True, duration, usage_info)

    # PROCESAR TODAS LAS FILAS Y ACUMULAR ESTADÍSTICAS Y TOKEN USAGE
    def _process_all_rows(chain: LLMChain, sheet: gspread.Worksheet) -> None:
        records = params["source_table_df"].to_dict(orient="records")
        row_range = params.get("source_table_row_range", "all")
        if isinstance(row_range, str) and row_range.lower() == "all":
            data_list = records
        elif isinstance(row_range, str) and "-" in row_range:
            start_row, end_row = map(int, row_range.split("-"))
            data_list = records[start_row - 1: end_row]
        else:
            idx = int(row_range) - 1
            data_list = [records[idx]]
        total_rows = len(data_list)
        processed_count = 0
        skipped_count = 0
        total_time = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_all_tokens = 0

        print(f"\n [RANGE INFO ℹ️] Total de filas a procesar: {total_rows}", flush=True)
        for i, row in enumerate(data_list, start=1):
            if i > 1 and params.get("system_prompt_second_and_later"):
                new_prompt_text = params["system_prompt_second_and_later"]
                chain.prompt.messages[0] = SystemMessagePromptTemplate.from_template(new_prompt_text, template_format="jinja2")
                print(f"[PROMPT INFO ℹ️] Para la fila {i} se usa system_prompt_second_and_later.", flush=True)

            processed, duration, usage = _process_row_and_write(chain, sheet, row, i)
            if processed:
                processed_count += 1
                total_time += duration
                total_prompt_tokens += usage.get("prompt_tokens", 0)
                total_completion_tokens += usage.get("completion_tokens", 0)
                total_all_tokens += usage.get("total_tokens", 0)
            else:
                skipped_count += 1

        avg_time = total_time / processed_count if processed_count else 0
        avg_prompt = total_prompt_tokens / processed_count if processed_count else 0
        avg_completion = total_completion_tokens / processed_count if processed_count else 0
        avg_total = total_all_tokens / processed_count if processed_count else 0

        print("\n🔹🔹🔹 [FINISHED ✅] Resumen de procesamiento: 🔹🔹🔹", flush=True)
        print(f"  - Filas totales: {total_rows}", flush=True)
        print(f"  - Filas procesadas: {processed_count}", flush=True)
        print(f"  - Filas omitidas: {skipped_count}", flush=True)
        print(f"  - Tiempo total LLM: {round(total_time, 2)} s", flush=True)
        print(f"  - Tiempo promedio por fila: {round(avg_time, 2)} s", flush=True)
        print("\n[TOKENS USAGE SUMMARY ℹ️]", flush=True)
        print(f"  - prompt_tokens totales: {total_prompt_tokens}", flush=True)
        print(f"  - completion_tokens totales: {total_completion_tokens}", flush=True)
        print(f"  - total_tokens totales: {total_all_tokens}", flush=True)
        print(f"  - prompt_tokens promedio: {round(avg_prompt, 2)}", flush=True)
        print(f"  - completion_tokens promedio: {round(avg_completion, 2)}", flush=True)
        print(f"  - total_tokens promedio: {round(avg_total, 2)}", flush=True)
        print("\n🔹🔹🔹 [FINISHED ✅] Proceso completado.", flush=True)

    print("🔹🔹🔹 [START ▶️] Iniciando LLM_process_text 🔹🔹🔹","\n", flush=True)
    try:
        _validate_params()
        print("[VALIDATION SUCCESS ✅] Todos los parámetros han sido validados.", flush=True)
        # Si se define la clave para obtener LLM_API_key desde Secret Manager, se recupera aquí:
        _retrieve_llm_api_key_from_secret_manager()
        sheet = _auth_with_google_sheets()
        _append_sheet_header(sheet)
        system_prompt_text = _get_system_prompt_text()
        chain = _configure_llm_chain(system_prompt_text)
        _process_all_rows(chain, sheet)
        print("🔹🔹🔹 [FINISHED ✅] LLM_process_text finalizado. Resultados escritos en Google Sheets.", flush=True)
    except ValueError as ve:
        print(f"[ERROR ❌] Error de validación: {ve}", flush=True)
        raise
    except Exception as ex:
        print(f"[ERROR ❌] Error inesperado: {ex}", flush=True)
        raise
