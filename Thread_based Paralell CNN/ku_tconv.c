#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>
pthread_mutex_t mutex1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t mutex2 = PTHREAD_MUTEX_INITIALIZER;

typedef struct {    // cworker 와 pworker에서 공통으로 사용하는 정보

    int infd;
    off_t offset;
    int input_size;
    int crs;    // convolution result size
    int** convRes;
    int** poolRes;

}info;

typedef struct {

    int pt_num;     // 생성된 pthread 번호
    int* remained_cworkers;     // 남은 cworker 개수
    info* info;

}conv_arg;

typedef struct {

    int pt_num;     // 생성된 pthread 번호
    int* remained_pworkers;     // 남은 pworker 개수
    info* info;

}pool_arg;


int get_inputsize(int infd);    // input file 로 부터 input array의 크기 받는 함수

void* convolution(void* argument);      // cworker 스레드 함수

void make_cworkers(void* convolution(void*), conv_arg* cargs, int cwork_count, int* remained_cworkers, info* info);      // cworker 생성 함수

void* pooling(void* argument);      // pworker 스레드 함수

void make_pworkers(void* pooling(void*), pool_arg* pargs, int pwork_count, int* remained_pworkers, info* info);      // pworker 생성 함수

void write_result(int convResSize, int** poolRes, int outfd);   // output file에 결과를 write 하는 함수



int main(int argc, char** argv) {

    int size = 0;   // input 크기
    int convResSize;    // conv 결과 크기
    int cwork_count;
    int pwork_count;

    int infd = open(argv[1], O_RDONLY);
    int outfd = open(argv[2], O_CREAT | O_RDWR | O_TRUNC, S_IRWXU);

    size = get_inputsize(infd);     // 첫줄의 개행 문자까지 읽어서 input 크기 확인


    off_t offset = lseek(infd, 0, SEEK_CUR);    // 현재 offset

    convResSize = size - 3 + 1;
    cwork_count = convResSize * convResSize;      // cworker 개수
    pwork_count = convResSize * convResSize / 4;    // pworker 개수


    // conv 결과 배열
    int** convRes = (int**)malloc(sizeof(int*) * convResSize);
    for (int i = 0; i < convResSize; i++)
        convRes[i] = (int*)malloc(sizeof(int) * convResSize);
    // pooling 결과 배열
    int** poolRes = (int**)malloc(sizeof(int*) * convResSize / 2);
    for (int i = 0; i < convResSize / 2; i++)
        poolRes[i] = (int*)malloc(sizeof(int) * convResSize / 2);


    // cworker와 pworker에서 사용할 정보
    info info;
    info.infd = infd;
    info.input_size = size;
    info.crs = convResSize;
    info.offset = offset;
    info.convRes = convRes;
    info.poolRes = poolRes;
    int* remained_cworkers = malloc(sizeof(int));   // cworker 들끼리 공유할 수 있도록 동적항당
    int* remained_pworkers = malloc(sizeof(int));   // pworker 들끼리 공유할 수 있도록 동적할당
    *remained_cworkers = cwork_count;
    *remained_pworkers = pwork_count;

    conv_arg* cargs = malloc(sizeof(conv_arg) * cwork_count);
    pool_arg* pargs = malloc(sizeof(pool_arg) * pwork_count);


    // cworkers 생성 -> 병렬로 convolution 연산
    make_cworkers(convolution, cargs, cwork_count, remained_cworkers, &info);

    while (1)   // 아직 처리가 끝나지 않은 cworker 가 있는지 확인
        if (*remained_cworkers == 0)
            break;

    // pworkers 생성 -> 병렬로 pooling 연산
    make_pworkers(pooling, pargs, pwork_count, remained_pworkers, &info);

    while (1)   // 아직 처리가 끝나지 않은 pworker 가 있는지 확인
        if (*remained_pworkers == 0)
            break;


    // write pooling result
    write_result(convResSize, poolRes, outfd);


    // free 및 close 
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


// input file 로 부터 input array의 크기 받는 함수
int get_inputsize(int infd) {

    char rbuf[4];
    char* bufp = rbuf;
    int size = 0;

    while (1) {             // 첫줄의 개행 문자까지 읽어서 input 크기 확인
        read(infd, bufp, 1);
        if (*rbuf == '\n')
            break;
        size = size * 10 + atoi(rbuf);
    }

    return size;
}


// convolution 스레드 함수
void* convolution(void* argument) {

    conv_arg* args = argument;

    // 읽어야할 위치(offset) 계산
    int offset = args->info->offset + (args->pt_num / args->info->crs) * 3 * args->info->input_size + (args->pt_num % (args->info->crs)) * 3;
    int convRessize = args->info->crs;
    int i = args->pt_num / convRessize;
    int j = args->pt_num % convRessize;
    int result = 0;

    int filter[3][3] = { { -1, -1, -1 }, { -1, 8, -1 }, { -1, -1, -1 } };
    int input[3][3] = { { 0, 0, 0 }, { 0, 0, 0 }, { 0, 0, 0 } };
    char rbuf[3];
    char* bufp = rbuf;

    for (int i = 0; i < 3; i++) {    // 3*3 input 읽기
        for (int j = 0; j < 3; j++) {
            pread(args->info->infd, bufp, 3, offset);
            input[i][j] = atoi(rbuf);
            offset += 3;
        }
        offset += 3 * (args->info->input_size - 3);
    }

    for (int i = 0; i < 3; i++)       // conv 연산 수행
        for (int j = 0; j < 3; j++)
            result += input[i][j] * filter[i][j];

    args->info->convRes[i][j] = result; // 결과 저장
    pthread_mutex_lock(&mutex1);
    (*args->remained_cworkers)--;   // 남은 cworker 개수 1 감소시키고 스레드 종료
    pthread_mutex_unlock(&mutex1);

}


// cworkers 생성 함수
void make_cworkers(void* (*convolution)(void*), conv_arg* cargs, int cwork_count, int* remained_cworkers, info* info) {

    pthread_t pid;

    for (int i = 0; i < cwork_count; i++) {
        cargs[i].pt_num = i;
        cargs[i].info = info;
        cargs[i].remained_cworkers = remained_cworkers;

        pthread_create(&pid, NULL, convolution, (void*)&cargs[i]);
        pthread_detach(pid);    // 작업이 끝난 스레드는 알아서 해제됨
    }

}


// pooling 스레드 함수
void* pooling(void* argument) {

    pool_arg* args = argument;
    int size = args->info->crs / 2;

    // max pooling 연산 수행
    int max = args->info->convRes[(2 * args->pt_num / args->info->crs) * 2][2 * args->pt_num % args->info->crs];
    int j = 0;
    while (j++ < 3)
        if (args->info->convRes[(2 * args->pt_num / args->info->crs) * 2 + j / 2][2 * args->pt_num % args->info->crs + j % 2] > max)
            max = args->info->convRes[(2 * args->pt_num / args->info->crs) * 2 + j / 2][2 * args->pt_num % args->info->crs + j % 2];

    args->info->poolRes[args->pt_num / size][args->pt_num % size] = max;    // 결과 저장
    pthread_mutex_lock(&mutex2);
    (*args->remained_pworkers)--;   // 남은 pworker 개수 1 감소시키고 스레드 종료
    pthread_mutex_unlock(&mutex2);

}


// pworkers 생성 함수
void make_pworkers(void* (*pooling)(void*), pool_arg* pargs, int pwork_count, int* remained_pworkers, info* info) {

    pthread_t pid;

    for (int i = 0; i < pwork_count; i++) {
        pargs[i].pt_num = i;
        pargs[i].info = info;
        pargs[i].remained_pworkers = remained_pworkers;

        pthread_create(&pid, NULL, pooling, (void*)&pargs[i]);
        pthread_detach(pid);    // 작업이 끝난 스레드는 알아서 해제됨
    }

}


// output file에 결과를 write 하는 함수
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