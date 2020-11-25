#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

typedef struct {    // cworker �� pworker���� �������� ����ϴ� ����

    int infd;
    off_t offset;
    int input_size;
    int crs;    // convolution result size
    int** convRes;
    int** poolRes;

}info;

typedef struct {

    int pt_num;     // ������ pthread ��ȣ
    int* remained_cworkers;     // ���� cworker ����
    info* info;

}conv_arg;

typedef struct {

    int pt_num;     // ������ pthread ��ȣ
    int* remained_pworkers;     // ���� pworker ����
    info* info;

}pool_arg;


int get_inputsize(int infd);    // input file �� ���� input array�� ũ�� �޴� �Լ�

void* convolution(void* argument);      // cworker ������ �Լ�

void make_cworkers(void* convolution(void*), conv_arg* cargs, int cwork_count, int* remained_cworkers, info* info);      // cworker ���� �Լ�

void* pooling(void* argument);      // pworker ������ �Լ�

void make_pworkers(void* pooling(void*), pool_arg* pargs, int pwork_count, int* remained_pworkers, info* info);      // pworker ���� �Լ�

void write_result(int convResSize, int** poolRes, int outfd);   // output file�� ����� write �ϴ� �Լ�



int main(int argc, char** argv) {

    int size = 0;   // input ũ��
    int convResSize;    // conv ��� ũ��
    int cwork_count;
    int pwork_count;

    int infd = open(argv[1], O_RDONLY);
    int outfd = open(argv[2], O_CREAT | O_RDWR | O_TRUNC, S_IRWXU);

    size = get_inputsize(infd);     // ù���� ���� ���ڱ��� �о input ũ�� Ȯ��


    off_t offset = lseek(infd, 0, SEEK_CUR);    // ���� offset

    convResSize = size - 3 + 1;
    cwork_count = convResSize * convResSize;      // cworker ����
    pwork_count = convResSize * convResSize / 4;    // pworker ����


    // conv ��� �迭
    int** convRes = (int**)malloc(sizeof(int*) * convResSize);
    for (int i = 0; i < convResSize; i++)
        convRes[i] = (int*)malloc(sizeof(int) * convResSize);
    // pooling ��� �迭
    int** poolRes = (int**)malloc(sizeof(int*) * convResSize / 2);
    for (int i = 0; i < convResSize / 2; i++)
        poolRes[i] = (int*)malloc(sizeof(int) * convResSize / 2);


    // cworker�� pworker���� ����� ����
    info info;
    info.infd = infd;
    info.input_size = size;
    info.crs = convResSize;
    info.offset = offset;
    info.convRes = convRes;
    info.poolRes = poolRes;
    int* remained_cworkers = malloc(sizeof(int));   // cworker �鳢�� ������ �� �ֵ��� �����״�
    int* remained_pworkers = malloc(sizeof(int));   // pworker �鳢�� ������ �� �ֵ��� �����Ҵ�
    *remained_cworkers = cwork_count;
    *remained_pworkers = pwork_count;

    conv_arg* cargs = malloc(sizeof(conv_arg) * cwork_count);
    pool_arg* pargs = malloc(sizeof(pool_arg) * pwork_count);


    // cworkers ���� -> ���ķ� convolution ����
    make_cworkers(convolution, cargs, cwork_count, remained_cworkers, &info);

    while (1)   // ���� ó���� ������ ���� cworker �� �ִ��� Ȯ��
        if (*remained_cworkers == 0)
            break;

    // pworkers ���� -> ���ķ� pooling ����
    make_pworkers(pooling, pargs, pwork_count, remained_pworkers, &info);

    while (1)   // ���� ó���� ������ ���� pworker �� �ִ��� Ȯ��
        if (*remained_pworkers == 0)
            break;


    // write pooling result
    write_result(convResSize, poolRes, outfd);


    // free �� close 
    free(cargs);
    free(pargs);
    free(remained_cworkers);
    free(remained_pworkers);

    for (int i = 0; i < convResSize; i++)
        free(convRes[i]);
    free(convRes);

    for (int i = 0; i < convResSize / 2; i++)
        free(poolRes[i]);
    free(poolRes);

    close(infd);
    close(outfd);

    pthread_mutex_destroy(&mutex1);
    pthread_mutex_destroy(&mutex2);
}


// input file �� ���� input array�� ũ�� �޴� �Լ�
int get_inputsize(int infd) {

    char rbuf[4];
    char* bufp = rbuf;
    int size = 0;

    while (1) {             // ù���� ���� ���ڱ��� �о input ũ�� Ȯ��
        read(infd, bufp, 1);
        if (*rbuf == '\n')
            break;
        size = size * 10 + atoi(rbuf);
    }

    return size;
}


// convolution ������ �Լ�
void* convolution(void* argument) {

    conv_arg* args = argument;

    // �о���� ��ġ(offset) ���
    int offset = args->info->offset + (args->pt_num / args->info->crs) * 3 * args->info->input_size + (args->pt_num % (args->info->crs)) * 3;
    int convRessize = args->info->crs;
    int i = args->pt_num / convRessize;
    int j = args->pt_num % convRessize;
    int result = 0;

    int filter[3][3] = { { -1, -1, -1 }, { -1, 8, -1 }, { -1, -1, -1 } };
    int input[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
    char rbuf[3];
    char* bufp = rbuf;

    for (int i = 0; i < 3; i++) {    // 3*3 input �б�
        for (int j = 0; j < 3; j++) {
            pread(args->info->infd, bufp, 3, offset);
            input[i][j] = atoi(rbuf);
            offset += 3;
        }
        offset += 3 * (args->info->input_size - 3);
    }

    for (int i = 0; i < 3; i++)       // conv ���� ����
        for (int j = 0; j < 3; j++)
            result += input[i][j] * filter[i][j];

    args->info->convRes[i][j] = result; // ��� ����
    pthread_mutex_lock(&mutex1);
    (*args->remained_cworkers)--;   // ���� cworker ���� 1 ���ҽ�Ű�� ������ ����
    pthread_mutex_unlock(&mutex1);

}


// cworkers ���� �Լ�
void make_cworkers(void* (*convolution)(void*), conv_arg* cargs, int cwork_count, int* remained_cworkers, info* info) {

    pthread_t pid;

    for (int i = 0; i < cwork_count; i++) {
        cargs[i].pt_num = i;
        cargs[i].info = info;
        cargs[i].remained_cworkers = remained_cworkers;

        pthread_create(&pid, NULL, convolution, (void*)&cargs[i]);
        pthread_detach(pid);    // �۾��� ���� ������� �˾Ƽ� ������
    }

}


// pooling ������ �Լ�
void* pooling(void* argument) {

    pool_arg* args = argument;
    int size = args->info->crs / 2;

    // max pooling ���� ����
    int max = args->info->convRes[(2 * args->pt_num / args->info->crs) * 2][2 * args->pt_num % args->info->crs];
    int j = 0;
    while (j++ < 3)
        if (args->info->convRes[(2 * args->pt_num / args->info->crs) * 2 + j / 2][2 * args->pt_num % args->info->crs + j % 2] > max)
            max = args->info->convRes[(2 * args->pt_num / args->info->crs) * 2 + j / 2][2 * args->pt_num % args->info->crs + j % 2];

    args->info->poolRes[args->pt_num / size][args->pt_num % size] = max;    // ��� ����
    pthread_mutex_lock(&mutex2);
    (*args->remained_pworkers)--;   // ���� pworker ���� 1 ���ҽ�Ű�� ������ ����
    pthread_mutex_unlock(&mutex2);

}


// pworkers ���� �Լ�
void make_pworkers(void* (*pooling)(void*), pool_arg* pargs, int pwork_count, int* remained_pworkers, info* info) {

    pthread_t pid;

    for (int i = 0; i < pwork_count; i++) {
        pargs[i].pt_num = i;
        pargs[i].info = info;
        pargs[i].remained_pworkers = remained_pworkers;

        pthread_create(&pid, NULL, pooling, (void*)&pargs[i]);
        pthread_detach(pid);    // �۾��� ���� ������� �˾Ƽ� ������
    }

}


// output file�� ����� write �ϴ� �Լ�
void write_result(int convResSize, int** poolRes, int outfd) {

    char rbuf[4];
    char* bufp = rbuf;

    for (int i = 0; i < convResSize / 2; i++) {
        for (int j = 0; j < convResSize / 2; j++) {
            sprintf(rbuf, "%d", poolRes[i][j]);

            for (int k = 0; k < 4 - strlen(rbuf); k++)
                write(outfd, " ", 1);
            write(outfd, rbuf, strlen(rbuf));

            if (j == convResSize / 2 - 1)
                write(outfd, "\n", 1);
            else
                write(outfd, " ", 1);
        }
    }

}