# Reconocimiento de Ojos con Redes Neuronales

Este proyecto implementa un sistema de **reconocimiento de ojos** en tiempo real utilizando **Deep Learning** con **PyTorch**. El sistema es capaz de identificar a personas a partir de las características únicas de sus ojos.

<img width="1200" height="493" alt="image" src="https://github.com/user-attachments/assets/ef5347f8-80aa-4135-95e3-e4e6e17ca308" />


## Instalación

### Requisitos:

* **Python 3.8+**
* **pip** (para gestionar las dependencias)
* **Torch** (PyTorch)
* **OpenCV** (para procesamiento de imágenes)
* **NumPy** (para cálculos rápidos)
* **pyttsx3** (para alertas de voz)

### Pasos para la instalación:

1. **Clonar el Repositorio:**
    ```bash
    git clone https://github.com/EdD4rk/Reconocimiento-de-Ojos-con-Python.git
    cd Reconocimiento-de-Ojos-con-Python
    ```

2. **Instalar `pip`** (si no está instalado):
    Si no tienes **pip** instalado, descárgalo y ejecútalo con el siguiente comando:
    ```bash
    python get-pip.py
    ```

3. **Instalar las dependencias**:
    Una vez tengas `pip` instalado, ejecuta los siguientes comandos para instalar todas las dependencias necesarias: `torch`,`opencv`,`numpy`,`pyttsx3`
    ```bash
    pip install torch
    pip install opencv-python
    pip install numpy
    pip install pyttsx3
    ```

4. **Ejecutar el proyecto**:
    Finalmente, ejecuta el script **`reconocimiento_camara.py`** para iniciar el sistema de reconocimiento en tiempo real:
    ```bash
    python reconocimiento_camara.py
    ```

    Puedes ver el sistema en acción directamente en tu cámara web local.

---

### Funcionalidades del Sistema de Reconocimiento de Ojos

```bash
☑ Identificación de personas por características únicas de los ojos  
☑ Alertas de voz para identificar personas detectadas  
☑ Capacidad de reconocer múltiples ojos en una sola imagen  
☒ Reconocimiento en ángulos extremos (por mejorar en futuras versiones)
