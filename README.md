### Descripción del Proyecto

Este proyecto fue desarrollado por **Davalos Bautista Lizbeth** y **Mendoza Ayma Edwin**, estudiantes de **Ingeniería de Sistemas e Informática** de la **Facultad de Ingeniería** de la **Universidad Tecnológica de los Andes**, como parte del curso de **Big Data**.  

El sistema de **reconocimiento de ojos** utiliza **Deep Learning** para identificar a las personas a partir de sus ojos en tiempo real, superando las limitaciones del reconocimiento facial tradicional. Implementado con **PyTorch** y **OpenCV**, el proyecto permite detectar ojos a través de una cámara web, crear un **embedding** único de cada ojo y comparar con una base de datos para realizar la identificación. Además, el sistema incluye **alertas de voz** que avisan cuando se detecta una persona conocida.

Con este proyecto, se busca explorar y aplicar tecnologías avanzadas de **Big Data** y **Reconocimiento Biométrico**, ofreciendo una solución práctica para **autenticación sin contacto** y **seguridad**.

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
```

### Demostracion:

<img width="1365" height="715" alt="Captura de pantalla 2025-08-25 143852" src="https://github.com/user-attachments/assets/833b7399-e219-4443-8f7f-3c906574e12c" />
