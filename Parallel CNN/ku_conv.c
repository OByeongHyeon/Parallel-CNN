#include <stdio.h>
#include <stdlib.h>
#include <mqueue.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>
#include <signal.h>
#define NAME "/m_queue"
#define NAME2 "/m_queue2"

void makeMatrix(int** matrix, int X, int Y);
void makeCworkers(int count, mqd_t mqdes, mqd_t mqdes2);
void insertConvInputtoQ(mqd_t mqdes, int** inputMtx, int numofmsg, int convResSize);
void makePworkers(int count, mqd_t mqdes, mqd_t mqdes2);
void receiveConvRes(mqd_t mqdes, int** convRes, int numofmsg, int convResSize);
void insertPoolInputtoQ(mqd_t mqdes, int** convRes, int numofmsg, int convResSize);
void receiveFinalRes(mqd_t mqdes, int* result, int numofmsg);
void handler(int sig);



int main(int argc, char** argv) {

    signal(SIGCHLD, handler);

    int size = atoi(argv[1]);   // input Matrix size
    int convResSize = size - 3 + 1;  // Convolutional Layer �� output Matrix size 
    int numofmsg = convResSize * convResSize; // conv ������ ������ 3*3 Matrix ����


    // input Matrix ���� 
    int** inputMtx = (int**)malloc(sizeof(int*) * size);
    for (int i = 0; i < size; i++) {
        inputMtx[i] = (int*)malloc(sizeof(int) * size);
    }

    makeMatrix(inputMtx, size, size);


    // convolution ����� ������ �迭 
    int** convRes = (int**)malloc(sizeof(int*) * convResSize);
    for (int i = 0; i < convResSize; i++)
        convRes[i] = (int*)malloc(sizeof(int) * convResSize);

    // Final output�� ������ �迭
    int* result = (int*)malloc(sizeof(int) * numofmsg / 4);


    // message queue ����
    struct  mq_attr attr;
    attr.mq_maxmsg = 10;
    attr.mq_msgsize = 36;

    struct  mq_attr attr2;
    attr2.mq_maxmsg = 10;
    attr2.mq_msgsize = 4;

    mqd_t mqdes = mq_open(NAME, O_CREAT | O_RDWR, 0600, &attr);
    if (mqdes < 0) {
        perror("mq_open()");
        exit(0);
    }

    mqd_t mqdes2 = mq_open(NAME2, O_CREAT | O_RDWR, 0600, &attr2);
    if (mqdes2 < 0) {
        perror("mq_open()");
        exit(0);
    }

    // numofmsg ���� child process �����Ͽ� convolution ���� ����
    makeCworkers(numofmsg, mqdes, mqdes2);


    // input Matrix�� Filter ũ�⸸ŭ �����Ͽ� Message queue�� ����    
    insertConvInputtoQ(mqdes, inputMtx, numofmsg, convResSize);


    // message queue���� convolution ����� �޾� convRes �迭�� ����
    receiveConvRes(mqdes2, convRes, numofmsg, convResSize);


    // numofmsg/4 ���� child process �����Ͽ� Max-pooling ���� ����
    makePworkers(numofmsg / 4, mqdes, mqdes2);


    // max_pooling 2x2 input�� message queue�� �ִ� ����
    insertPoolInputtoQ(mqdes, convRes, numofmsg / 4, convResSize);


    // message queue ���� Max-pooling ����� �޾� Result �迭�� ����
    receiveFinalRes(mqdes2, result, numofmsg / 4);


    // Final output ���
    for (int i = numofmsg / 4 - 1; i > 0; i--) {
        printf("%d ", result[i]);
    }
    printf("%d", result[0]);

    // �����Ҵ� �迭 Free
    for (int i = 0; i < size; i++) {
        free(inputMtx[i]);
    }
    free(inputMtx);

    for (int i = 0; i < convResSize; i++) {
        free(convRes[i]);
    }
    free(convRes);
    free(result);

    mq_close(mqdes);
    mq_unlink(NAME);
    mq_close(mqdes2);
    mq_unlink(NAME2);

}


void makeCworkers(int count, mqd_t mqdes, mqd_t mqdes2) {

    int filter[3][3] = { { -1, -1, -1 }, { -1, 8, -1 }, { -1, -1, -1 } };
    int input[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
    unsigned int prio = 0;
    int result = 0;     // convolution ���� ���

    for (int i = 0; i < count; i++) {

        if (fork() == 0) {

            if (mq_receive(mqdes, (char*)input, sizeof(input), &prio) == -1) {
                perror("child mq_receive()");
            }

            // conv ���� ����
            for (int i = 0; i < 3; i++)
                for (int j = 0; j < 3; j++)
                    result += (input[i][j] * filter[i][j]);

            if (mq_send(mqdes2, (char*)&result, sizeof(result), prio) == -1) {
                perror("child mq_send()");
            }

            exit(0);
        }
    }
}

void insertConvInputtoQ(mqd_t mqdes, int** inputMtx, int numofmsg, int convResSize) {

    int arr[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };

    for (int i = 0; i < numofmsg; i++) {

        // Filter ũ�� ��ŭ ����
        for (int j = i / convResSize; j < i / convResSize + 3; j++) {
            for (int k = i % convResSize; k < i % convResSize + 3; k++) {
                arr[j - i / convResSize][k - i % convResSize] = inputMtx[j][k];
            }
        }

        // message queue �� ������ ��� ����
        if (mq_send(mqdes, (char*)arr, sizeof(arr), numofmsg - i) == -1) {
            perror("mq_send()");
        }
    }
}

void receiveConvRes(mqd_t mqdes, int** convRes, int numofmsg, int convResSize) {

    unsigned int prio = 0;
    int tmp = 0;

    for (int i = 0; i < numofmsg; i++) {

        if (mq_receive(mqdes, (char*)&tmp, 4, &prio) == -1) {
            perror("mq_receive()");
        }

        convRes[(numofmsg - prio) / convResSize][(numofmsg - prio) % convResSize] = tmp;
    }

}

void makePworkers(int count, mqd_t mqdes, mqd_t mqdes2) {

    int input[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
    unsigned int prio = 0;

    for (int i = 0; i < count; i++) {
        if (fork() == 0) {

            if (mq_receive(mqdes, (char*)input, sizeof(input), &prio) == -1) {
                perror("mq_receive()");
            }

            // Max-pooling ���� ����
            int max = input[0][0];
            int j = 0;
            while (j++ < 3)
                if (input[j / 2][j % 2] > max)
                    max = input[j / 2][j % 2];


            if (mq_send(mqdes2, (char*)&max, sizeof(max), prio) == -1) {
                perror("mq_send()");
            }

            exit(0);
        }
    }
}

void insertPoolInputtoQ(mqd_t mqdes, int** convRes, int numofmsg, int convResSize) {

    unsigned int prio = numofmsg;
    int arr[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };

    for (int i = 0; i < convResSize; i += 2) {
        for (int j = 0; j < convResSize; j += 2) {
            for (int k = 0; k < 2; k++)
                for (int l = 0; l < 2; l++)
                    arr[k][l] = convRes[i + k][j + l];

            if (mq_send(mqdes, (char*)arr, sizeof(arr), prio--) == -1) {
                perror("max_pooling mq_send()");
            }
        }
    }
}

void receiveFinalRes(mqd_t mqdes, int* result, int numofmsg) {

    unsigned int prio = 0;
    int tmp = 0;

    for (int i = 0; i < numofmsg; i++) {
        if (mq_receive(mqdes, (char*)&tmp, sizeof(tmp), &prio) == -1) {
            perror("mq_receive()");
        }
        result[prio - 1] = tmp;
    }
}

void handler(int sig) {

    pid_t pid;

    while ((pid = waitpid(-1, NULL, WNOHANG)) > 0) {
        // printf("%d process reaping\n", pid);
    }
    // if (errno != ECHILD)
    //     sio_error("waitpid error");
}