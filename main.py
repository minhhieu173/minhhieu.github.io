#import streamlit as st

#st.title("chao mung ban den voi streamlit")
#st.write("ung dung web dau tien cua ban bang python")

import streamlit as st

# Tiêu đề ứng dụng
st.title("Form Đăng Ký Thông Tin")

# Tạo form với Streamlit
with st.form("form_dang_ky"):
    ho_ten = st.text_input("Họ và tên")
    email = st.text_input("Email")
    mat_khau = st.text_input("Mật khẩu", type="password")
    submit = st.form_submit_button("Đăng ký")

    if submit:
        if not ho_ten or not email or not mat_khau:
            st.warning("Vui lòng điền đầy đủ thông tin.")
        else:
            # Xử lý lưu hoặc xác thực tại đây (ghi file, lưu DB, ...)
            st.success(f"Đăng ký thành công cho {ho_ten}!")
