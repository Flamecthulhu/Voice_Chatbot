#include <windows.h>

LRESULT CALLBACK WindowProcedure(HWND, UINT, WPARAM, LPARAM);

int main() {
    const char className[] = "WindowClass";
    HINSTANCE hInstance = GetModuleHandle(NULL);

    // 註冊視窗類別
    WNDCLASSEX wc = { sizeof(WNDCLASSEX), CS_CLASSDC, WindowProcedure, 0L, 0L, hInstance, NULL, NULL, NULL, NULL, className, NULL };
    RegisterClassEx(&wc);

    // 創建無邊框視窗
    HWND hwnd = CreateWindowEx(
        WS_EX_APPWINDOW,              // 擴展樣式
        className,                    // 視窗類別名稱
        "Frameless Window",           // 視窗標題
        WS_POPUP,                     // 無邊框樣式
        CW_USEDEFAULT, CW_USEDEFAULT, // 視窗位置QA
        800, 600,                     // 視窗大小
        NULL, NULL, hInstance, NULL   // 其他設置
    );

    ShowWindow(hwnd, SW_SHOWDEFAULT);  // 顯示視窗
    UpdateWindow(hwnd);                // 更新視窗

    // 消息循環
    MSG msg;
    while (GetMessage(&msg, NULL, 0, 0)) {
        TranslateMessage(&msg);
        DispatchMessage(&msg);
    }

    return 0;
}

LRESULT CALLBACK WindowProcedure(HWND hwnd, UINT msg, WPARAM wParam, LPARAM lParam) {
    switch (msg) {
        case WM_DESTROY:
            PostQuitMessage(0);
            break;
        default:
            return DefWindowProc(hwnd, msg, wParam, lParam);
    }
    return 0;
}

