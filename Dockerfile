# parameters
ARG ARCH
ARG NAME
ARG ORGANIZATION
ARG DESCRIPTION
ARG MAINTAINER

# ==================================================>
# ==> Do not change the code below this line
ARG BASE_REGISTRY=docker.io
ARG BASE_ORGANIZATION=ripl
ARG BASE_REPOSITORY=sofa_base
ARG BASE_TAG=main

# define base image
FROM ${BASE_REGISTRY}/${BASE_ORGANIZATION}/${BASE_REPOSITORY}:${BASE_TAG}-${ARCH} as BASE

# recall all arguments
# - current project
ARG NAME
ARG ORGANIZATION
ARG DESCRIPTION
ARG MAINTAINER
# - base project
ARG BASE_REGISTRY
ARG BASE_ORGANIZATION
ARG BASE_REPOSITORY
ARG BASE_TAG
# - defaults
ARG LAUNCHER=default

# define/create project paths
ARG PROJECT_PATH="${CPK_SOURCE_DIR}/${NAME}"
ARG PROJECT_LAUNCHERS_PATH="${CPK_LAUNCHERS_DIR}/${NAME}"
RUN mkdir -p "${PROJECT_PATH}"
RUN mkdir -p "${PROJECT_LAUNCHERS_PATH}"
WORKDIR "${PROJECT_PATH}"

# keep some arguments as environment variables
ENV \
    CPK_PROJECT_NAME="${NAME}" \
    CPK_PROJECT_DESCRIPTION="${DESCRIPTION}" \
    CPK_PROJECT_MAINTAINER="${MAINTAINER}" \
    CPK_PROJECT_PATH="${PROJECT_PATH}" \
    CPK_PROJECT_LAUNCHERS_PATH="${PROJECT_LAUNCHERS_PATH}" \
    CPK_LAUNCHER="${LAUNCHER}"

# install apt dependencies
COPY ./dependencies-apt.txt "${PROJECT_PATH}/"
RUN cpk-apt-install ${PROJECT_PATH}/dependencies-apt.txt

# install python3 dependencies
COPY ./dependencies-py3.txt "${PROJECT_PATH}/"
RUN cpk-pip3-install ${PROJECT_PATH}/dependencies-py3.txt

# install launcher scripts
COPY ./launchers/. "${PROJECT_LAUNCHERS_PATH}/"
COPY ./launchers/default.sh "${PROJECT_LAUNCHERS_PATH}/"
RUN cpk-install-launchers "${PROJECT_LAUNCHERS_PATH}"

# copy project root
COPY ./*.cpk ./*.sh ${PROJECT_PATH}/

# copy the source code
COPY ./packages "${CPK_PROJECT_PATH}/packages"

# build catkin workspace
RUN catkin build \
    --workspace ${CPK_CODE_DIR}

# define default command
CMD ["bash", "-c", "launcher-${CPK_LAUNCHER}"]

# store module metadata
LABEL \
    cpk.label.current="${ORGANIZATION}.${NAME}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.description="${DESCRIPTION}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.code.location="${PROJECT_PATH}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.base.registry="${BASE_REGISTRY}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.base.organization="${BASE_ORGANIZATION}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.base.project="${BASE_REPOSITORY}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.base.tag="${BASE_TAG}" \
    cpk.label.project.${ORGANIZATION}.${NAME}.maintainer="${MAINTAINER}"
# <== Do not change the code above this line
# <==================================================


COPY ./assets/py3.pth /usr/lib/python3/dist-packages/packages.pth
COPY ./assets/py2.pth /usr/lib/python2.7/dist-packages/packages.pth

ENV DISPLAY :0
COPY ./assets/dummy.conf /dummy.conf
RUN python3.7 -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu \
    -f https://download.pytorch.org/whl/torch_stable.html
RUN python3.7 -m pip install pyautogui==0.9.41
RUN python2 -m pip install --no-cache-dir pynput pyautogui

COPY ./assets/mor_edits/sceneCreation.py /builds/python/mor/utility/sceneCreation.py
COPY ./assets/mor_edits/phase1_snapshots.py /builds/python/mor/reduction/template/phase1_snapshots.py
COPY ./assets/mor_edits/ReadGieFileAndComputeRIDandWeights.py /builds/python/mor/reduction/script/ReadGieFileAndComputeRIDandWeights.py
COPY ./assets/mor_edits/reduceModel.py /builds/python/mor/reduction/reduceModel.py
COPY ./assets/mor_edits/reductionParam.py /builds/python/mor/reduction/container/reductionParam.py
COPY ./assets/plugin_list.conf.default /builds/sofa/lib/plugin_list.conf.default
