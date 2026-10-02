# PROMPT DUY NHẤT CHO ANTIGRAVITY — HOÀN THIỆN DATA MINING 01–07 THEO RUBRIC

Bạn là kỹ sư Data Science chịu trách nhiệm triển khai, đồng thời là người phản biện phương pháp và kiểm định khả năng tái lập. Hãy trực tiếp hoàn thiện dự án dưới đây trong workspace hiện có. Tôi cần code, notebook, thực nghiệm, output, dashboard và hồ sơ báo cáo hoàn chỉnh; không chỉ cần một bản nhận xét hoặc kế hoạch.

Branch mục tiêu: `feature-insights`.

Phạm vi: toàn bộ project, đặc biệt:

1. `notebooks/01_data_understanding.ipynb`
2. `notebooks/02_eda_and_cleaning.ipynb`
3. `notebooks/03_feature_engineering_rfm.ipynb`
4. `notebooks/04_customer_clustering.ipynb`
5. `notebooks/05_repeat_purchase_classification.ipynb`
6. `notebooks/06_association_rules.ipynb`
7. `notebooks/07_model_comparison_and_insights.ipynb`

Yêu cầu cốt lõi: với MỖI quyết định phân tích quan trọng, người đọc phải trả lời được **làm gì — tại sao làm — căn cứ ở đâu — so với phương án nào — thông số và kết quả thực tế là gì — kết luận được đến đâu**. Không chỉ bổ sung vài đoạn Markdown chung chung vào notebook.

## 0. Cách làm việc và giới hạn

- Kiểm tra branch, commit, trạng thái worktree và dữ liệu có sẵn trước khi sửa. Không reset, xóa thay đổi của người dùng, tự chuyển nhánh làm mất công việc, tự commit/push/merge, phát hành release hoặc deploy công khai.
- Nếu code hiện tại mới hơn snapshot bên dưới, đọc lại và kiểm chứng từng vấn đề. Không cố “sửa” một lỗi đã được xử lý.
- Đọc đầy đủ notebook cả source lẫn output, runner, module, tests, app, tài liệu, rubric trong repo nếu có, cấu hình và output thực tế. Không suy luận “hoàn thiện” chỉ vì có đủ tên file.
- Lập kế hoạch và audit ngắn, sau đó TRIỂN KHAI — TEST — CHẠY THỰC NGHIỆM — VIẾT GIẢI THÍCH — NGHIỆM THU. Không dừng ở kế hoạch khi còn có thể thực hiện công việc trong phạm vi này.
- Giữ code tốt đang có; refactor vừa đủ để loại bỏ logic trùng giữa runner/notebook. Không viết lại toàn bộ hoặc thêm mô hình phức tạp chỉ để tăng số lượng.
- Không bịa kết quả, output, benchmark, nguồn trích dẫn, biên bản góp ý, tên thành viên, đóng góp hay peer assessment. Dữ liệu tổng hợp chỉ dùng cho unit test, không làm kết quả chính.
- Không hard-code kết quả để khớp báo cáo cũ, không ép K=2/RandomForest/FP-Growth thắng, không đổi tiêu chí sau khi nhìn test để có kết quả đẹp.
- Nếu thiếu raw data: dùng nguồn công khai chính thức nếu môi trường cho phép; không thay thế âm thầm bằng sample hoặc dataset khác. Nếu bị chặn, hoàn thiện phần code/test có thể làm, báo rõ phần chưa chạy và dữ liệu cần người dùng cung cấp. Không tuyên bố pipeline PASS.
- Dữ kiện do con người cung cấp còn thiếu được ghi `NEEDS_USER_INPUT`, không ngăn cản sửa phần kỹ thuật độc lập. Những phần này không được đánh dấu hoàn thiện.
- Dùng tiếng Việt có dấu cho phần diễn giải; giữ thuật ngữ và tên API khi cần. Thống nhất đơn vị tiền tệ GBP/£, không đổi thành USD/$.

## 1. Snapshot audit cần kiểm chứng trước khi sửa

Các quan sát sau dựa trên source và output ĐÃ LƯU ở commit `b58a7427e950799c52d31b28b8c624923546774d`. Đây không phải chứng nhận đã chạy lại toàn bộ pipeline trên raw data. Các con số chỉ là điểm đối chiếu lịch sử, không phải “expected results” để ép chương trình đạt.

### 1.1 Những phần đã có, cần giữ và phát triển

- Đã có đủ notebook 01–07, runner từng bước, module `src/`, tests cơ bản và app Streamlit.
- Bước 04 đã chạy K-Means, GMM, Agglomerative và DBSCAN; đã có nhiều chỉ số và logic xếp hạng.
- Bước 05 có Dummy, Logistic Regression, Decision Tree, Random Forest; preprocessing bằng Pipeline và CV trên train.
- Bước 03 đã tách feature period và label window; cần kiểm chứng toàn diện hơn, không nói rằng dự án hoàn toàn chưa xử lý leakage.
- Notebook 07 hiện đã có execution count cho 10/10 code cells và output. Không tiếp tục nhận xét theo phiên bản cũ rằng notebook 07 chưa chạy.
- `run_pipeline.py` hiện đã xử lý việc thiếu/0-byte output bằng kết quả thất bại; vấn đề còn lại là độ đầy đủ, đúng nội dung và tính nhất quán của bộ kiểm tra.

### 1.2 Các vấn đề cụ thể phải đưa vào danh sách sửa

| ID | Vị trí | Quan sát cần xử lý |
|---|---|---|
| A01 | Notebook 01 | Output lưu vẫn có ImportError liên quan `FIGURES_DATA_UNDERSTANDING`; phần lớn code cells không có execution count. Cấu hình hiện tại đã có biến này, nên phải phân biệt traceback cũ với lỗi còn tái hiện; chạy lại clean kernel để chứng minh. |
| A02 | Notebook 06 | Output lưu có OSError khi ghi vào thư mục lồng `association_rules/association_rules`; 3 code cells cuối không có execution count dù vẫn còn output cũ. |
| A03 | Notebook 02, 03, 07 | Một số phần chỉ hiển thị PNG có sẵn bằng `if exists`; không tái tạo biểu đồ và có thể bỏ qua file thiếu. Notebook 07 gần như chỉ có một Markdown mở đầu, còn thiếu diễn giải phản biện. |
| A04 | Runner/notebook 01 | `Total Columns` lấy kích thước DataFrame sau khi thêm cột; một số chú thích phân vị/phần trăm viết cố định; nhận diện StockCode bắt đầu bằng chữ chưa đồng nghĩa mã phi sản phẩm. |
| A05 | Preprocessing và docs | Tuyên bố missing CustomerID chắc chắn là khách vãng lai; duplicate chắc chắn do gửi nhiều lần; mọi outlier là đơn bán buôn — vượt quá bằng chứng. Kiểm tra raw chỉ bằng shape không chứng minh toàn vẹn nội dung. |
| A06 | `src/feature_engineering.py` | Kiểm tra hướng điểm Recency trong fallback của `_safe_qcut` và ảnh hưởng ties/thứ tự dòng. Tham số horizon có thể thay đổi nhưng tên target/validator còn gắn cứng 90 ngày. |
| A07 | Basket generation | Đếm `Description` để tạo basket có thể bỏ sót item thiếu mô tả; ép chuỗi trước dropna có thể giữ chuỗi “nan”; ước lượng memory boolean nhưng tạo int gây lệch dự toán. |
| A08 | Clustering | Có rank tổng hợp nhưng chưa xuất đầy đủ rank/lý do loại/tiêu chí; bảng upstream lưu `is_selected=False` cho tất cả. Tên “Lost Customers” chưa được chứng minh là churn. |
| A09 | Classification | Chưa có quy trình tìm kiếm siêu tham số thực sự. CV F1 lưu: Dummy khoảng 0.72594, RF 0.67258, LR 0.66498; code vẫn chọn RF sau khi loại Dummy. Phải đánh giá giá trị so với baseline trung thực, không gọi RF là tốt nhất mọi mặt. |
| A10 | Prediction artifacts | `test_predictions.csv` không giữ CustomerID; khó truy vết sai số và join đúng khách hàng. `runtime_seconds` classification hiện không phải toàn bộ thời gian huấn luyện. |
| A11 | Notebook/runner 06 | Notebook nói FP-Growth nhanh hơn dù output của chính notebook ghi Apriori 1.97s, FP-Growth 2.13s. Báo cáo 07 lại dùng runtime từ lượt khác. Runner 06 cố định lưu selected rules từ FP-Growth; bước 07 có thể chọn thuật toán khác theo runtime. |
| A12 | `src/association_rules.py` | `valid_lift` được tính nhưng không thay thế tập trả về; nhiều check chỉ in “[OK]” thay vì thực sự enforce. Xóa mọi luật có conviction vô hạn và thay conviction thiếu bằng 0 cần sửa theo định nghĩa toán học. |
| A13 | `src/model_comparison.py` | Hàm so sánh rules ghi “element-by-element” nhưng không đọc cả hai bộ luật để đối chiếu. Số lượng bằng nhau không chứng minh nội dung bằng nhau. |
| A14 | Artifact verification | Kiểm tra classification chỉ xét `is_selected` trong khi upstream dùng `selected`, nên có thể bỏ qua đối chiếu. Kiểm tra load được Pipeline chưa chứng minh đúng estimator, tham số, feature schema và threshold. |
| A15 | `src/insights.py` | Join full-history clusters/RFM với nhãn repeat theo cutoff cũ; mean bỏ qua nhãn thiếu nhưng customer_count lấy toàn bộ nhóm. Cần công khai thời gian, mẫu số và giới hạn diễn giải. |
| A16 | Insight/report 07 | Có đoạn strategic findings hard-code số nhóm, tỷ trọng, repeat rate, importance, số luật. Một số logic ngưỡng nghiệp vụ chưa có căn cứ; phần feature importance gọi cả biến rất thấp là “strong association”. |
| A17 | Plot 07 | `_plot_clustering_quality` lọc “KMeans” trong khi upstream dùng “K-Means”, có thể xuất PNG có dung lượng nhưng trống dữ liệu. Biểu đồ còn giả định chỉ có 2 cụm, tiền tệ dùng $. |
| A18 | Provenance | Summary được commit trên branch hiện tại nhưng trường git_commit là `006a149...`; cần ghi đúng HEAD tại thời điểm chạy và dirty-state/source digest, không kết luận chỉ từ việc khác SHA rằng kết quả chắc chắn sai. Manifest hiện có thể bỏ qua output thiếu mà vẫn PASS. |
| A19 | Docs/app/tests | README mô tả một số file/chức năng chưa có thực tế; app hiện chủ yếu upload CSV và predict classification. Team tasks còn là mẫu phân công. Chưa có báo cáo cuối CRISP-DM đầy đủ hoặc bộ kiểm thử đủ mạnh cho clustering, rules và pipeline. |

Chú ý: dữ liệu, phần lớn CSV/PNG và model đang bị gitignore. Không thấy chúng trên GitHub không có nghĩa người dùng chưa tạo; nhưng manifest hoặc lời báo “đã có” không thay thế được việc kiểm tra file thực tế và khả năng tái tạo.

## 2. Rubric đích và ma trận minh chứng

Hoàn thiện theo 6 tiêu chí sau, hướng đến mức Xuất sắc nhưng không tự bảo đảm điểm giảng viên:

| Tiêu chí | Trọng số | Minh chứng cần có |
|---|---:|---|
| Business understanding | 10% | Bài toán thực tế, người dùng kết quả, quyết định được hỗ trợ, mục tiêu khai phá chuyển từ mục tiêu nghiệp vụ; đối chiếu góp ý tiến độ nếu có dữ kiện thật. |
| Data understanding & preprocessing | 20% | Thống kê, missing, outlier, phân bố lớp; giải thích quyết định và đo tác động tiền xử lý lên kết quả mô hình, không chỉ số dòng bị loại. |
| Lựa chọn & xây dựng mô hình | 25% | So sánh phản biện ít nhất 2–3 phương pháp phù hợp; ưu/nhược, giả định, chi phí, suitability; tuning có quy trình và kết quả. |
| Đánh giá & phân tích kết quả | 20% | Metric phù hợp, validation/CV đúng, baseline, sai số, ý nghĩa thực tiễn, giới hạn suy luận. |
| Kỹ thuật & tái lập | 15% | Code module hóa, môi trường rõ, pipeline chạy lại được, tests, notebook có output mới, demo trực quan và bằng chứng chạy. |
| Báo cáo & tổng kết | 10% | Báo cáo CRISP-DM thống nhất số liệu; đóng góp thực tế/peer assessment; hạn chế trung thực, thông tin sử dụng AI. |

Tạo `docs/rubric_evidence_matrix.md`, mỗi hàng có: criterion/subrequirement, notebook/section hoặc cell-id, code/function, output/table/figure, test/log, trạng thái `PASS/PARTIAL/FAIL/BLOCKED`, việc còn thiếu. PASS phải dẫn tới minh chứng đã kiểm tra.

Không nhầm 7 notebook với 6 phase CRISP-DM. Không áp điều kiện của Research Track A sang Track B. Đọc rubric gốc nếu có: không tự thêm yêu cầu Final phải thuyết trình/slide khi quy định Final chỉ yêu cầu report, code và minh chứng demo. Vẫn chuẩn bị nội dung giải trình khi được hỏi, nhưng không làm slide như một điều kiện bắt buộc nếu không được yêu cầu.

## 3. Chuẩn bắt buộc cho mọi notebook và mọi quyết định

### 3.1 Khuôn diễn giải cho mỗi khối phân tích quan trọng

Mỗi bước tạo cột, lọc dữ liệu, biến đổi, chọn tham số, chọn model, chọn metric, join dữ liệu, lập insight phải có đủ:

1. **Câu hỏi/mục đích:** khối này trả lời điều gì, liên hệ câu hỏi nghiệp vụ nào?
2. **Đầu vào và đơn vị phân tích:** file/hash/version, tập khách hàng/hóa đơn/dòng sản phẩm, khoảng thời gian, số mẫu.
3. **Cách làm:** công thức, định nghĩa, điều kiện, tham số, đơn vị.
4. **Lý do và căn cứ:** từ tài liệu chính thức, đặc tính dữ liệu đo được, thực nghiệm hay giả định nghiệp vụ? Tách rõ bốn loại.
5. **Phương án đối chứng:** lựa chọn đơn giản hơn/khác là gì, tại sao giữ hoặc bỏ? Quyết định có thể đo tác động phải có thí nghiệm, không chỉ lý thuyết.
6. **Kết quả thực chạy:** bảng/biểu đồ, số mẫu, ngưỡng, metric, mean/std hoặc mức bất định thích hợp; dẫn output cụ thể.
7. **Giải thích kết quả:** mô tả bằng số; giải thích cơ chế hợp lý nhưng phân biệt điều quan sát được với giả thuyết về nguyên nhân.
8. **Kết luận và hạn chế:** chọn gì, mất gì, chưa chứng minh được gì, chuyển sang bước nào?

Không cần viết luận cho từng dòng import hoặc `mkdir`. Nhưng mọi quyết định ảnh hưởng dữ liệu, mô hình, cách đánh giá và kết luận đều phải giải thích được. Mỗi biểu đồ có nhận xét ngay sau, không chỉ “nhìn vào biểu đồ”.

### 3.2 Từ điển cột và nhật ký quyết định

Tạo `docs/feature_decision_dictionary.md` và `docs/decision_log.md`:

- Mỗi cột được thêm: tên, nguồn, công thức, cấp độ dòng/invoice/customer, datatype, đơn vị, thời điểm sẵn có, xử lý thiếu, mục đích, rủi ro leakage, thực sự có đưa vào model nào không.
- Phân biệt cột phục vụ EDA, QA flag, khóa join, feature mô hình, target, audit-only. Không đưa tất cả cột mới vào mô hình.
- Bao gồm cột thời gian và cờ trong 01–02; TotalAmount; R/F/M; TotalItems; UniqueProducts; ActiveDays; AverageOrderValue; AverageItemsPerInvoice; CustomerLifetimeDays; Country; RFM score/segment; log features; cluster; nhãn tương lai và các cột audit.
- Nếu giữ `UniqueInvoices` cùng `Frequency`, nêu chúng có trùng nghĩa không, vì sao không đưa cả hai vào model.
- Mỗi quyết định có decision_id, phương án, lý do, evidence_id, mã nguồn thực thi và trạng thái.

Ví dụ về mức giải thích cần đạt: “Tạo TotalAmount để đo giá trị của một dòng hàng bằng Quantity × UnitPrice, đơn vị GBP; tổng theo InvoiceNo là giá trị hóa đơn trong phạm vi giao dịch đã định nghĩa. Đây là giá trị mua hàng, không tự động là lợi nhuận hoặc doanh thu ròng. Cột hỗ trợ EDA và tính Monetary/AOV, không phải biến khách hàng dùng trực tiếp trước khi aggregate. Đối chiếu phép tính trên một số giao dịch thật và test công thức.” Không chép ví dụ này như thể đã thực nghiệm nếu chưa chạy.

### 3.3 Benchmark, ngưỡng và nguồn

- Benchmark chính: baseline trên CHÍNH dữ liệu, cohort, split/folds và protocol của project.
- Benchmark thứ hai: ablation/sensitivity, thay một quyết định có kiểm soát để đánh giá tác động.
- Kết quả trong bài báo ngoài chỉ là đối chiếu tham khảo nếu không cùng dataset version, cohort, nhãn, window và split. Không lấy một F1/Silhouette ở bài khác làm chuẩn bắt buộc.
- Không có cơ sở thì không nói “Silhouette > 0.5 luôn tốt”, “F1 phải > 0.8”, “K=2 là chuẩn”, “support 0.02 là tối ưu”, “class_weight balanced luôn tốt”.
- Tham số mặc định/heuristic phải được nhận diện đúng và có sensitivity phù hợp; không bịa nguồn học thuật cho quyết định riêng của nhóm.
- Tra cứu nguồn gốc/official docs phù hợp phiên bản đang chạy. Ghi nguồn vào `docs/references.md` hoặc BibTeX: tác giả/tổ chức, tên, URL/DOI đã xác minh, mục được dùng. Notebook dẫn nguồn ở đúng đoạn ra quyết định.
- Các nguồn khởi đầu để kiểm tra: UCI Online Retail; scikit-learn clustering, CV, model evaluation, grid search, common pitfalls, threshold tuning; tài liệu mlxtend Apriori/FP-Growth/association_rules. Đọc nguồn liên quan, không chỉ liệt kê URL.

### 3.4 Sổ thực nghiệm và kết luận tự cập nhật

- Lưu bảng thực nghiệm máy đọc được với experiment_id, run_id, input hash, feature_version, cohort/cutoff/window, split_id, fold/seed, model/config_id, toàn bộ tham số, metric, runtime, status và lý do lỗi/N/A.
- Chỉ làm tròn khi trình bày, không chọn model từ số liệu đã làm tròn.
- Mọi con số trong caption, summary, Markdown report và dashboard lấy từ kết quả hiện tại hoặc được đối chiếu tự động. Không chép cố định “2 cụm”, “61 luật”, “95.7%”, “RF tốt nhất”.
- Một output hợp lệ không chỉ là tồn tại và >0 byte: phải đọc được, đúng schema, đúng run/provenance, metric hợp lệ theo định nghĩa và biểu đồ có dữ liệu.
- Khi không thể chứng minh tại sao một mẫu hình phát sinh, ghi “giả thuyết giải thích, cần kiểm chứng”; không biến correlation/importance/co-occurrence thành nhân quả.

## 4. Hoàn thiện notebook 01 — Data understanding

### Việc phải làm

- Xác minh nguồn UCI, ý nghĩa 8 cột thực tế, grain “một dòng hàng trong hóa đơn”, thời gian, currency, provenance và SHA256 của raw file. Phân biệt số feature trong metadata website với tổng số cột của file.
- Nếu metadata nguồn mâu thuẫn với dữ liệu thực đọc được, ví dụ vấn đề missing, nêu rõ khác biệt; không bỏ qua missing vì website nói không có.
- Thống kê dòng, hóa đơn, sản phẩm, khách hàng, quốc gia; kiểu dữ liệu; missing theo cột và nhóm; duplicate trên bộ cột gốc; ngày thiếu/parse lỗi.
- Kiểm tra cùng InvoiceNo có bất nhất CustomerID/Country/time không, cùng StockCode có nhiều mô tả không; lượng hóa thay vì giả định khóa luôn sạch.
- Phân tích invoice cancellation, adjustment, quantity/price không dương, mã phí/dịch vụ/quà tặng/sản phẩm đặc biệt. Prefix C có căn cứ nguồn; prefix A và các mã đặc biệt phải đối chiếu record thực tế, không khái quát vượt dữ liệu.
- Phân phối Quantity, UnitPrice, TotalAmount: median, quantile, skewness, extreme records; phân biệt raw, positive-purchase subset và clean subset.
- EDA thời gian/tháng, top sản phẩm/quốc gia và mức tập trung khách hàng/doanh số; ghi đúng mẫu số. Chú thích tháng quan sát chưa đủ bằng ngày thực tế.
- Không diễn giải CustomerID như biến số liên tục có mean mang ý nghĩa hành vi.
- Giải thích mọi cột phục vụ EDA: InvoicePrefix/Month/Hour nếu có, IsCancelled, IsAdjust, TotalAmount… Cột không dùng thì không tạo cho đủ.
- Sửa số cột raw bị tăng do thêm feature; bỏ số liệu caption hard-code. Nếu zoom/cắt P99 hoặc ẩn outlier để đọc hình, ghi rõ phần dữ liệu không hiển thị và kèm thống kê toàn bộ.
- Top product aggregate theo StockCode chuẩn; Description là nhãn hiển thị, không làm tách một sản phẩm thành nhiều nhóm không chủ ý.
- Cuối notebook: vấn đề dữ liệu → số lượng/tỷ lệ → bằng chứng → cách xử lý dự kiến ở 02 → tác động/rủi ro.

### Output tối thiểu

`raw_metadata.json`, raw summary, schema/data dictionary, missing summary, duplicate summary, invoice/stockcode audit, bảng nguồn của các biểu đồ, PNG và output inline. Đặt trong stage directory phù hợp; có thể giữ tên file sẵn có và mở rộng.

## 5. Hoàn thiện notebook 02 — EDA and cleaning

### 5.1 Làm sạch có thể giải trình

- Bảo toàn raw; kiểm tra hash trước/sau, không chỉ shape.
- Chuẩn hóa khóa và kiểu dữ liệu có báo cáo giá trị không parse được. Không đổi lỗi số thành 0 âm thầm, không biến missing ID thành chuỗi “nan”, không tự tạo khách hàng giả.
- Với mỗi quy tắc: bằng chứng raw, định nghĩa, thứ tự áp dụng, số dòng bị tác động, số hóa đơn/khách bị ảnh hưởng, giá trị mua hàng thay đổi, trade-off và lý do.
- Giữ audit các dòng loại hoặc thống kê có khả năng truy vết; tách số cờ lỗi có thể chồng lấn khỏi số dòng bị loại tuần tự. Tổng before − removed = after phải khớp.
- Xác minh chính sách exact duplicate: khi nào tính trên 8 cột gốc, có bị thay đổi sau impute Description không? Duplicate có thể phản ánh ghi nhận lặp, nhưng nguyên nhân không biết chắc; ghi giả định và đo sensitivity thích hợp.
- Missing Description: mapping theo StockCode, rule xử lý mode hòa, số điền bằng mapping và số còn “Unknown Product”. “Không còn NaN” không có nghĩa “khôi phục chính xác 100% mô tả gốc”.
- Missing CustomerID: giữ cho phân tích hóa đơn/sản phẩm khi phù hợp; loại khỏi aggregation cấp khách hàng và giải thích selection bias. So sánh nhóm có/thiếu ID theo thời gian, quốc gia, basket/value; không khẳng định toàn bộ là khách vãng lai.
- Cancellation/adjustments/nonpositive values: phân biệt thống kê giao dịch thô, giao dịch mua hợp lệ và giá trị sau hoàn/hủy. Nếu chỉ loại dòng hủy chứ không đối soát hoàn trả với hóa đơn gốc, không gọi Monetary là net customer value.
- StockCode phi sản phẩm: bảng mapping có record minh chứng; không xóa mọi mã bắt đầu bằng chữ. Quyết định cho POST, DOT, M/m, C2, D, S, BANK CHARGES, AMAZONFEE, CRUK, B, PADS, DCGS*, gift_* phải nhất quán với mục tiêu phân tích, có mục “không xác định” khi thiếu căn cứ.
- Ba nhánh `cleaned_transactions`, `customer_transactions`, `product_transactions` phải có định nghĩa rõ và bảng parent/child. Product branch không phải tiếp tục lọc từ customer branch nếu cần giữ hóa đơn thiếu CustomerID.
- Đặt đúng tên file `missing_values_after_cleaning`: phải phản ánh đúng giai đoạn được gọi, không lấy số sau impute nhưng trước toàn bộ cleaning rồi mô tả là cuối pipeline.

### 5.2 Outlier và bằng chứng tác động

- IQR = Q3 − Q1, fences theo hệ số cấu hình; giải thích đây là heuristic phát hiện, không chứng minh điểm đó sai.
- Chỉ ra phân phối lệch, các record cực trị và quy mô bán buôn có thể là một nguyên nhân. Không mặc định mọi điểm bị flag đều hợp lệ.
- So sánh phương án có kiểm soát: giữ giá trị hợp lệ + biến đổi log/scale; một lựa chọn robust/quantile clipping hợp lý. Không reintroduce giao dịch lỗi để tạo baseline kém.
- Threshold học từ phân phối dùng trong classification phải fit trong train/fold, không từ cả test. EDA mô tả toàn bộ dữ liệu phải tách rõ khỏi transformer dùng trong modeling.
- Bắt buộc có đánh giá tác động downstream: metric CV classification và chất lượng/độ ổn định clustering cho các phương án liên quan, kèm số mẫu/khách hàng giữ lại.
- Thiết kế thí nghiệm ở 02, thực thi phần mô hình ở 04–05, tổng hợp ở 07; tránh tạo vòng phụ thuộc 02 bắt buộc chạy 05 trước. Bản notebook/báo cáo nộp cuối dẫn tới kết quả downstream của đúng run.
- Nếu phương án “phức tạp hơn” không cải thiện, chấp nhận và chọn phương án đơn giản dựa trên chứng cứ.

### Output tối thiểu

Ba dataset interim; cleaning ledger có before/removed/after/parent; missing/duplicate/stockcode/outlier audits; trước–sau với cùng định nghĩa dân số; bảng thiết kế sensitivity và kết quả downstream được liên kết; toàn bộ biểu đồ được sinh từ hàm dùng chung và hiển thị inline.

## 6. Hoàn thiện notebook 03 — Feature engineering and RFM

### 6.1 RFM và biến bổ sung

- Giải thích vì sao customer-level, Frequency là unique InvoiceNo chứ không phải số dòng.
- Định nghĩa R/F/M, reference_date, độ chính xác thời gian, units; cộng đối chiếu tổng Monetary về đúng tập nguồn.
- Giải thích quy ước `reference_date = last_observation + 1 day`; đó là quy ước tính khoảng thời gian, không phải điều kiện toán học bắt buộc vì `log1p(0)` vẫn xác định.
- Mỗi feature mở rộng có giả thuyết hành vi, công thức và nguy cơ dư thừa. Không dùng từ “Lifetime” như dự đoán CLV; giải thích CustomerLifetimeDays hiện là thời gian quan sát giữa lần đầu/cuối trong cửa sổ.
- Dùng vài khách hàng thật để đối chiếu từ transaction → invoice → RFM/behavioral features bằng phép tính dễ kiểm tra; không bịa ví dụ như kết quả dataset.
- Kiểm tra khóa duy nhất, missing theo hợp đồng, nonnegative/positive domains, Frequency–UniqueInvoices, AOV = Monetary/Frequency; tránh double log hoặc scale hai lần.
- Sửa scoring ties/fallback: R nhỏ phải cho điểm cao, F/M lớn điểm cao; xử lý ít unique/ít khách; cùng giá trị không bị phân chia tùy tiện theo thứ tự dòng nếu không có giải thích. Có regression tests kiểm tra hướng, ties và permutation invariance.
- Giải thích tại sao chọn 5 bins, cách xử lý khi không đủ bins; đây là phân khúc heuristic, không phải ground truth để chứng minh K-Means đúng. Nếu so heuristic với clusters, gọi là phân tích mức tương ứng.

### 6.2 Nhãn mua lại và point-in-time correctness

- Phân biệt bảng RFM toàn kỳ phục vụ mô tả với bảng feature tại cutoff phục vụ prediction.
- In rõ feature_start, cutoff, feature_reference_date, label_start/end, observation_end, horizon, cohort rule.
- Feature chỉ từ dữ liệu được biết đến cutoff; nhãn từ giao dịch hợp lệ trong `(cutoff, cutoff + horizon]`.
- Khách xuất hiện sau cutoff không có lịch sử: báo số lượng và loại khỏi cohort prediction này, không gán nhãn 0 vì không có feature.
- Chỉ gán “không mua lại” khi có đủ thời gian quan sát horizon; đánh dấu/loại right-censoring nếu có. Nêu giới hạn nếu thiếu chứng cứ hoạt động quan sát trong toàn cửa sổ.
- Duy trì horizon 90 ngày làm bài toán chính nếu hợp lý, giải thích bằng chu kỳ hành vi và mục đích marketing. Lượng hóa interpurchase intervals; cân nhắc 30/60/90 như sensitivity và nêu rõ thay horizon là thay target/cohort, không xếp hạng đơn giản bằng F1 khác bài toán.
- Nếu horizon configurable, tên target, metadata, validator, notebook, app và báo cáo phải đồng bộ. Không đổi tham số thành 30 nhưng vẫn gọi target 90d.
- Kiểm thử toàn bộ features quan trọng bằng aggregate chỉ trên prefix thời gian; append giao dịch tương lai không được làm đổi feature snapshot tại cutoff đã cố định. Kiểm tra ranh giới đúng bằng cutoff và label_end.
- Audit label có thể chứa future counts nhưng tuyệt đối không lọt vào X/model/app.
- Không tuyên bố “Recency >= 1 và không có hai cột future” là bằng chứng đủ cho toàn bộ pipeline không leakage.

### 6.3 Basket

- Một basket = một InvoiceNo; một item = StockCode; hiện diện boolean không phụ thuộc Description có thiếu hay Quantity > 1.
- Kiểm tra key trước ép kiểu, unique (InvoiceNo, StockCode), loại invoice/item rỗng theo chính sách được ghi.
- Giải thích vì sao không lấy Description làm khóa; mapping chỉ dùng hiển thị.
- Chọn dense boolean/sparse phù hợp thư viện, ước lượng memory theo dtype thực, ghi shape/sparsity/basket-size distribution và hóa đơn bị loại.
- Ma trận dựng từ long format phải khớp version đã lưu, có test equivalence.

### Output tối thiểu

RFM customer features, clustering features kèm danh sách THỰC DÙNG, repeat features, label audit riêng, cohort/cutoff metadata, feature summary/dictionary, class balance, basket long và representation dùng cho mining, các biểu đồ phân phối/correlation/log transform.

Đánh giá lợi ích feature theo nhóm ở 05: RFM-only so với RFM + behavior và thêm Country khi phù hợp. Không cần thí nghiệm riêng vô hạn cho từng cột; các biến không được chứng minh cải thiện phải diễn giải trung thực.

## 7. Hoàn thiện notebook 04 — Customer clustering

### 7.1 Lựa chọn phương pháp

- Giải thích clustering không có nhãn chuẩn, các mục tiêu tách nhóm/khả năng hành động và giới hạn internal metrics.
- So sánh K-Means, GMM, Ward Agglomerative, DBSCAN đang có: giả định hình dạng/khoảng cách, soft vs hard labels, noise, nhạy scale/outlier, tham số, chi phí tính toán và khả năng gán khách mới.
- Kiểm tra code thực dùng RFM hay features mở rộng; docs và metadata phải đúng. Không tuyên bố đã dùng các cột chỉ tồn tại trong CSV.
- Thực nghiệm ít nhất hai preprocessing representations hợp lý, ví dụ StandardScaler(RFM) và StandardScaler(log1p(RFM)); giải thích log làm giảm ảnh hưởng skew chứ không bảo đảm Gaussian.
- Internal metrics phụ thuộc không gian/khoảng cách: ghi rõ representation, không kết luận một transform “khách quan tốt hơn” chỉ vì silhouette ở geometry khác cao hơn. Kết hợp stability, profile và khả năng giải thích.

### 7.2 Trả lời bằng chứng “Tại sao chọn K=2?”

- Quét K=2..8 hiện có hoặc dải điều chỉnh được biện minh; ghi init, n_init, seed, max_iter và convergence. Elbow có thể gồm K=1 cho inertia, nhưng silhouette K=1 là N/A.
- Xuất đầy đủ config-level metrics: inertia/WCSS khi áp dụng, silhouette, Davies–Bouldin, Calinski–Harabasz, Dunn approximation nếu giữ, số cụm, cluster-size distribution, coverage/noise, runtime.
- Gọi đúng Dunn approximation và viết công thức thực đang dùng; không nhầm với Dunn index chính xác.
- In bảng ranking: điều kiện hợp lệ, lý do loại từng cấu hình, rank từng metric, trọng số/rank sum, tie-break và selected_config_id. Giải thích ngưỡng min size/max proportion là policy nghiệp vụ, không “chuẩn học thuật”.
- Không âm thầm nới điều kiện khi không còn candidate. Báo rõ phương án fallback và giới hạn; không chọn tự động silhouette max khi artifact/config không khớp.
- Tách metric trên non-noise khỏi tỷ lệ dữ liệu được bao phủ của DBSCAN. Cấu hình 1 cụm hoặc toàn noise phải được báo invalid/N/A, không bịa metric 0 hoặc crash cả benchmark.
- Kiểm tra chính xác units: noise_ratio là 0–1 hay 0–100; cluster percentages dùng đúng mẫu số.
- Phân tích K=2 cạnh K=3,4 và candidate cạnh tranh tốt nhất: chênh lệch số, độ cân bằng, profile, độ ổn định và nhu cầu business. WCSS giảm theo K không tự chứng minh K lớn tốt hơn; PCA đẹp không phải kiểm định.
- Với finalists của phương pháp ngẫu nhiên, dùng nhiều seed với budget hợp lý, chẳng hạn tối thiểu 5, ghi mean/std và ARI giữa các partition trên cùng CustomerIDs. Có thể thêm subsampling nếu khả thi; phân biệt seed stability với stability theo sample.
- Snapshot cũ có K-Means K=2 silhouette 0.4330; K=3 khoảng 0.3375, K=4 khoảng 0.3381; Ward K=2 khoảng 0.4232. DBSCAN eps=0.7/min=5 có silhouette cao hơn nhưng max_cluster_pct khoảng 99.33% và cụm nhỏ 5 khách. Hãy kiểm chứng lại và giải thích trade-off; không dùng các con số cũ thay kết quả mới.
- Nếu bằng chứng mới chọn K khác hoặc thuật toán khác, cập nhật mọi đầu ra. Nếu chọn K=2, kết luận phải đọc được ngay trong notebook với bảng minh chứng cụ thể và hạn chế của phân khúc chỉ hai nhóm.

### 7.3 Profile, artifacts và hình

- Profile bằng mean, median, quantile, quy mô và tỷ trọng giá trị mua; nêu ảnh hưởng khách cực lớn.
- Đặt tên mô tả như “gần đây/tần suất cao” hoặc “ít mua/lâu chưa quay lại” dựa dữ liệu. Không tự gán churn/lost chắc chắn khi không có định nghĩa churn hoặc quan sát đủ.
- PCA chỉ để trực quan nếu model fit trên RFM scaled; lưu explained variance đúng dạng metadata, không nhét tỷ lệ PC vào các dòng khách đầu tiên.
- Xuất elbow, metric-by-K, distribution, RFM profile và PCA có đúng population. Hình phải đọc được với mọi K hợp lệ, không chỉ K=2.
- Lưu transformer/feature list/model/config/labels/selection reason nhất quán. Dùng config_id duy nhất, không chỉ ghép algorithm+K nếu nhiều hyperparameters.
- Với thuật toán transductive không có predict cho khách mới, giải thích giới hạn; dashboard có thể tra cứu cụm đã gán, không giả vờ hỗ trợ predict trực tiếp.

### Output tối thiểu

Comparison đầy đủ, selection audit, stability/preprocessing sensitivity, profiles, customer_clusters, hình, model/scaler/config và provenance. Bảng comparison chỉ được có một selected configuration khi đã chọn, và phải khớp artifact.

## 8. Hoàn thiện notebook 05 — Repeat purchase classification

### 8.1 Protocol và mục tiêu đánh giá phải chốt trước

- Định nghĩa positive class, target horizon, đối tượng được dự báo và quyết định được hỗ trợ: ưu tiên liên hệ nhóm dễ mua lại hay tìm nhóm nguy cơ không mua? Hai mục tiêu không đồng nhất.
- Chọn primary metric theo mục tiêu, nêu trade-off FP/FN và secondary metrics. Nếu chưa có chi phí marketing/biên lợi nhuận, không bịa utility/ROI; ghi assumption hoặc đánh giá các kịch bản giả định rõ ràng.
- Giữ customer-level holdout, stratified folds khi mỗi khách một snapshot; lưu ID train/test/fold và seed. Không sort CustomerID rồi gọi đó là temporal split.
- Single-cutoff random customer split đánh giá khả năng phân biệt khách tại cùng giai đoạn, không chứng minh generalization sang thời gian tương lai.
- Nếu bổ sung temporal backtest: tạo snapshot/cutoff đúng thời gian, bảo đảm label window của tập train đã kết thúc trước thời điểm dự báo của tập đánh giá, kiểm soát khách lặp và overlap theo mục tiêu. Nếu dữ liệu không đủ, ghi limitation; không chế tạo thời gian hoặc bắt buộc TimeSeriesSplit trên bảng customer không có thứ tự thời gian hợp lệ.
- Holdout hiện đã được xem trong các vòng phát triển: công khai điều đó. Không tuyên bố đây là test hoàn toàn chưa từng được nhìn, không đổi seed nhiều lần để chọn test đẹp. Từ thời điểm chốt protocol, không dùng test để tune/chọn model/threshold.

### 8.2 Baseline, thuật toán, preprocessing

- Giữ Dummy baseline và 3 model LR/DT/RF là đủ nền tảng; không bắt buộc thêm XGBoost/SVM chỉ cho nhiều thuật toán.
- Giải thích ưu/nhược và suitability trên số khách, feature numeric/categorical, quan hệ phi tuyến, interpretability, chi phí và overfitting; không chỉ chép định nghĩa.
- Trình bày class balance thực tế, không mô tả 57/43 là “mất cân bằng nghiêm trọng” hoặc tự dùng SMOTE. Nếu thử class_weight, so None với balanced bằng CV.
- Pipeline chứa imputation, encoding, scaling/transform phù hợp model và toàn bộ bước học từ dữ liệu. Fit preprocessing trong mỗi fold; clone độc lập giữa models.
- Không cần chuẩn hóa numeric cho tree theo cùng lý do như LR; giải thích lựa chọn pipeline chung hoặc riêng và kiểm tra không làm sai dữ liệu.
- Phát hiện cột đưa vào nhưng bị ColumnTransformer silently drop; metadata phải phản ánh schema thực. Khóa ID và audit future chỉ dùng truy vết, không làm X.
- Benchmark majority/constant baseline phải được phân tích theo cả hai lớp. Khi positive là majority, predict toàn 1 có thể có positive-F1 cao nhưng specificity bằng 0.
- Xuất delta so với Dummy ở primary và secondary metrics. Có thể chọn một learned candidate để minh họa nhưng phải phân biệt “best learned candidate”, “best baseline” và “đủ giá trị triển khai hay chưa”.
- Tuyệt đối không xóa Dummy khỏi bảng, bỏ metric bất lợi hoặc đổi tiêu chí sau khi thấy test để khẳng định model thắng.

### 8.3 Tuning và feature/preprocessing ablation

- Thực hiện tìm kiếm siêu tham số bằng GridSearchCV/RandomizedSearchCV hoặc quy trình tương đương đúng phiên bản thư viện.
- Ghi search space, rationale, budget, số lần fit, folds, scoring/refit, random state và runtime.
- Ví dụ không gian bắt đầu có thể cân nhắc: LR C và class_weight; DT max_depth/min_samples_leaf/min_samples_split; RF n_estimators/max_depth/min_samples_leaf/max_features/class_weight. Giới hạn theo tài nguyên, ghi cấu hình thực tế; không coi các ví dụ là lựa chọn tối ưu đã biết.
- Lưu toàn bộ search results, best_params, fold scores mean/std và baseline-vs-tuned trên cùng protocol.
- Nếu dùng CV để vừa tune vừa tuyên bố chất lượng tổng quát, phân biệt search CV có selection bias với đánh giá độc lập. Dùng nested CV cho so sánh tuned variants nếu budget cho phép, hoặc có train/validation/holdout được tách rõ; không gọi best search score là ước lượng không thiên lệch.
- Bắt buộc thí nghiệm theo nhóm feature: RFM-only → RFM+behavior → cân nhắc Country. Giữ cohort/split/folds, tránh so model khác và features khác đồng thời rồi quy toàn bộ cải thiện cho một feature.
- Hoàn thiện thí nghiệm preprocessing/outlier của 02. Ghi rõ thay đổi nào làm đổi cohort; comparison chính dùng tập phù hợp để có thể diễn giải.
- Không triển khai tổ hợp vô hạn. Có experiment matrix trước khi chạy và ghi các nhánh bị cắt vì tài nguyên.

### 8.4 Threshold và đánh giá

- Threshold 0.5 là baseline, không tự động tối ưu nghiệp vụ.
- Nếu điều chỉnh threshold, dùng validation hoặc OOF predictions từ training. Nếu tuner/hyperparameter selection cũng dùng dữ liệu đó, ghi quy trình lồng/tách để tránh báo performance threshold quá lạc quan.
- Chọn threshold theo metric/constraint đã công bố; đóng băng cùng model trước khi xem test. Lưu threshold vào metadata và dùng chung trong CLI/notebook/app.
- Báo precision, recall, F1 theo lớp; positive F1, macro F1, accuracy, balanced accuracy, ROC-AUC, Average Precision, confusion matrix TN/FP/FN/TP và class prevalence.
- Gọi đúng Average Precision; không đổi nhãn thành diện tích PR tính hình thang nếu chưa tính metric đó. ROC baseline 0.5 và PR/prevalence reference phải được diễn giải đúng.
- Xuất ROC, PR, confusion matrix, CV mean/std, threshold trade-off, feature importance và phân tích lỗi có bảng nguồn.
- So train–validation/CV trên cùng metric để xem overfit, có learning curve hoặc sensitivity complexity phù hợp. CV–test chênh lệch không tự chứng minh overfit.
- Với bất định, dùng fold dispersion và/hoặc bootstrap khách hàng trên test để báo khoảng tin cậy mô tả, ghi cách tính và phạm vi. Không coi các folds phụ thuộc như các quan sát độc lập để tuyên bố ý nghĩa thống kê tùy tiện.
- Khi metric không xác định do thiếu một lớp hoặc cấu hình thất bại, lưu N/A + reason; không thay bằng số 0 như thể đo được.
- FP/FN analysis: CustomerID, y_true, y_pred, score/probability, threshold, đặc trưng tại cutoff; phân tích theo RFM/Country/quy mô nhóm. Bảo vệ thông tin nếu sau này có dữ liệu không công khai.
- Không dùng label train để báo hiệu quả như đánh giá độc lập. Nếu cần summary toàn cohort, dùng OOF cho train + test prediction đúng nguồn, hoặc phân biệt descriptive/in-sample rõ ràng.

### 8.5 Giải thích và lưu model

- Tree impurity importance là mức đóng góp theo model, có bias và chịu ảnh hưởng feature tương quan, không phải tỷ lệ tác động nhân quả.
- Giữ dấu hệ số LR nếu diễn giải chiều liên hệ; không lấy absolute rồi nói tăng/giảm.
- Cân nhắc permutation importance trên tập đánh giá phù hợp, có repeat/std và metric đi kèm; không sửa model sau khi khảo sát test.
- Nếu gọi score là xác suất đáng tin cậy cho quyết định, đánh giá calibration; calibration nếu thêm phải fit bằng train/validation đúng quy trình. Không mặc định predict_proba đã calibrated.
- Lưu pipeline đầy đủ + metadata + threshold + schema + model/config/split/run IDs; test load lại và predict ra cùng kết quả.
- Phân biệt fit time, tuning time, prediction latency và evaluation time; benchmark các model cùng điều kiện.
- Test predictions phải có CustomerID và không bị lệch index khi ghép X/y. Mọi baseline và candidate đều giữ trong comparison.

### Output tối thiểu

Dataset/cohort summary, split manifest, CV fold results, search results, best params, ablation results, model comparison, threshold analysis, classification reports, confusion matrices, test predictions có ID, error analysis, importance, figures, pipeline và metadata.

## 9. Hoàn thiện notebook 06 — Association rules

### 9.1 Định nghĩa và lựa chọn tham số

- Giải thích dùng Apriori và FP-Growth để giải cùng bài toán frequent itemsets; so cách tìm kiếm, chi phí theo số items/density/support và giới hạn triển khai. Không khẳng định FP-Growth luôn nhanh hơn.
- Dùng cùng basket, cùng min_support, các giới hạn độ dài nếu có và cùng rule filters. Không so runtime hai bài toán khác nhau.
- Định nghĩa support(A), support(B), support(A∪B), confidence A→B, lift, leverage và conviction nếu báo cáo. Xác định mẫu số là số hóa đơn/baskets, không phải dòng hàng hay số khách.
- Giải thích confidence bằng conditional co-occurrence trong dữ liệu này, không phải xác suất tác động khuyến mãi gây mua.
- Support 0.02/confidence 0.5/lift>1 phải có lý do, số hóa đơn tương ứng và sensitivity. Thử một lưới nhỏ khả thi quanh baseline, chẳng hạn support 0.01/0.02/0.03 và confidence 0.5/0.7; điều chỉnh theo tài nguyên và ghi rõ.
- Lưới lớn có thể gây bùng nổ itemsets; dự toán và log runtime/memory, không bắt buộc chạy cấu hình vượt khả năng máy.
- Không catch mọi exception rồi âm thầm tăng support. Bắt đúng loại lỗi, log và ghi cấu hình thực dùng; failed configuration vẫn có mặt trong audit.
- So số rules, support counts, coverage, antecedent length, độ trùng lặp/khả năng hành động; nhiều luật không tự động tốt hơn.

### 9.2 Xác thực đúng nội dung

- Lưu frequent itemsets và rules riêng cho cả hai thuật toán.
- Canonical key dùng StockCode sets được sắp xếp, serialization không mơ hồ (ví dụ JSON arrays); Description chỉ display. Không split tên sản phẩm bằng dấu phẩy để xác thực overlap.
- So equality bằng outer join canonical itemset/rule keys, rồi numeric comparison tolerance cho support/confidence/lift. Xuất only_in_apriori, only_in_fpgrowth, matched/mismatched và max metric delta.
- Chỉ nói “hai bộ kết quả tương đương trong tolerance” khi kiểm tra nội dung thực sự PASS. Cùng 389 itemsets/61 rules không đủ.
- Validator phải thật sự enforce nonempty antecedent/consequent, disjointness, uniqueness, support/confidence bounds, threshold filters và quan hệ metric. Không chỉ in “[OK]”.
- Các metric cốt lõi phải finite khi định nghĩa yêu cầu. Riêng conviction có thể +∞ hợp lệ khi confidence=1 và consequent chưa phổ biến 100%; không xóa luật hợp lệ vì vậy.
- Trường hợp conviction undefined phải được phân biệt với +∞ hợp lệ; không fill NaN bằng 0. Khi ghi JSON chuẩn dùng null + status/representation thích hợp, không xuất NaN/Infinity trái chuẩn.
- Nếu không có luật đạt điều kiện, xuất bảng rỗng ĐÚNG SCHEMA và reason; báo “không có luật khả dụng”, không bịa luật. Hợp đồng output phân biệt empty-valid-result với file thiếu/lỗi.

### 9.3 Benchmark và selection thống nhất

- Đo từng thuật toán nhiều lần, ví dụ 3–5 lần sau warm-up khi khả thi, ghi thứ tự/chế độ, cùng input và phần công việc được đo; báo median/dispersion, không suy rộng từ một lần chênh vài phần trăm.
- Tách frequent-itemset mining, rule generation và total nếu dùng cả ba. Không gọi một phần runtime là end-to-end.
- Chọn thuật toán từ chính so sánh này, hoặc nêu rõ lý do khác nếu runtime tương đương.
- `selected_association_rules`, cột algorithm, selection metadata, notebook 06, report 07, dashboard và manifest phải thống nhất. Không chỉ đổi nhãn tên thuật toán trên CSV của thuật toán khác.
- Chuẩn hóa thư mục `outputs/tables/association_rules/` và tạo parent rõ ràng; cập nhật mọi consumer, không vá riêng notebook để runner/app tiếp tục trỏ đường dẫn cũ.

### 9.4 Business interpretation

- Với mỗi luật tiêu biểu: StockCodes/names, basket count thực, support, confidence, base rate của consequent, lift/leverage, phạm vi dữ liệu và hạn chế.
- Lift 18 không có nghĩa “tăng doanh thu 18 lần”. Phân biệt relative association mạnh với bằng chứng yếu vì ít mẫu.
- Các nhãn Strong/Moderate/Weak phải có policy giải thích và không mâu thuẫn với câu khuyến nghị tự động.
- Đề xuất bundle/cross-sell là giả thuyết thử nghiệm; cần xét biên lợi nhuận, tồn kho, trùng sản phẩm, mùa vụ và kiểm chứng triển khai.
- Nếu có temporal validation rules khả thi, chọn/mining ở giai đoạn trước và đánh giá ở giai đoạn sau; không gọi random invoice split là kiểm chứng generalization theo thời gian. Nếu không làm, ghi rõ kết quả exploratory.

### Output tối thiểu

Basket audit, itemsets/rules hai thuật toán, quality validation, content-equivalence audit, parameter sensitivity, repeated runtime benchmark, selected rules + metadata, insights có số lượng nền và các biểu đồ.

## 10. Hoàn thiện notebook 07 — Model comparison and insights

Notebook 07 phải là báo cáo tổng hợp có lập luận, không phải một gallery CSV/PNG.

### 10.1 Đối chiếu kỹ thuật trước khi tổng hợp

- Đọc output/schema/metadata từ 01–06 của run hợp lệ; không trộn metric mới với hình/model cũ.
- So selected_config_id/model type/hyperparameters/feature schema/transformer/threshold giữa comparison, metadata và artifact thực load.
- Chuẩn hóa `selected`/`is_selected` có kiểm tra xung đột; boolean string “False” không được cast thành True.
- Nếu selection/artifact không khớp, fail rõ ràng; không chọn đại model theo silhouette để làm pipeline xanh.
- Rule equivalence phải dùng audit nội dung ở 06; nhắc lại counts không thay thế kiểm chứng.
- So classification CV/holdout provenance, sample counts và baseline deltas; phân biệt metric chính/phụ, best learned candidate và readiness.
- Tổng hợp ưu/nhược, tài nguyên, hạn chế từng phương pháp. Clustering, classification và rules giải ba bài toán khác nhau: không cộng F1 + silhouette + lift để tìm một “model tốt nhất toàn project”.

### 10.2 Sửa căn bản insight theo thời gian và mẫu số

- Mọi merge theo CustomerID cần normalize schema, unique-key checks, `validate` phù hợp và coverage report: matched, left-only, right-only.
- Hiện tại RFM/clusters dùng toàn kỳ còn nhãn repeat chỉ có cohort tại cutoff. Nếu giữ phép nối này, chỉ gọi là phân tích hồi cứu mô tả, vì cụm có thể đã dùng hành vi trong chính label window.
- Không dùng repeat rate chênh giữa hai cụm full-history làm bằng chứng độc lập rằng phân cụm dự báo tốt future repeat.
- Nếu muốn câu chuyện predictive theo segment, phải xây features/segmentation tại cutoff, học các bước liên quan trên phạm vi training phù hợp, gán evaluation customers đúng protocol và đánh giá nhãn tương lai độc lập.
- Với repeat rate mỗi nhóm phải xuất: total_customers, eligible_labeled_customers, repeat_customers, unlabeled_customers, coverage_pct và repeat_customers / eligible_labeled_customers; mẫu số 0 → N/A có lý do. Không fill nhãn thiếu bằng 0.
- Tỷ trọng khách/doanh số có thể dùng all-cohort, nhưng cần cột denominator/cohort riêng. Không ghép tỷ lệ 95.7% từ subset vào số lượng toàn nhóm rồi suy ra số khách mua lại.
- Label thực tế và predicted score là hai loại chứng cứ khác nhau; thể hiện rõ. Nếu ghép predictions, chỉ dùng OOF/holdout phù hợp và kiểm tra CustomerID.

### 10.3 Nội dung narrative bắt buộc

1. Executive summary: câu hỏi nghiệp vụ, dữ liệu/cohort, phát hiện đã kiểm chứng và giới hạn lớn nhất.
2. Chất lượng dữ liệu: mất bao nhiêu thông tin, bias gì, preprocessing ảnh hưởng kết quả ra sao.
3. Clustering: lý do chọn algorithm/K bằng bảng, rank và stability, không chỉ đọc lại tên winner.
4. Classification: lựa chọn theo CV/validation; baseline; tuned vs default; threshold; lỗi FP/FN; trường hợp chưa vượt baseline.
5. Association: correctness/equivalence, runtime thực nghiệm, độ phủ và chất lượng luật.
6. Phân khúc + repeat + rules: liên kết trong phạm vi thời gian hợp lệ; nếu chưa có segment-specific rules thì không trình bày global rule như thể đã được kiểm chứng riêng cho từng segment.
7. Action plan: nhóm đối tượng, metric chứng cứ, hành động đề xuất, KPI, phương án thử nghiệm, rủi ro/giới hạn.
8. Hạn chế và việc tiếp theo: single retailer, historical period, missing IDs, không có margin/campaign response, temporal drift, selection bias, chưa chứng minh nhân quả.

### 10.4 Hành động và biểu đồ

- Ngưỡng “recent <60”, “inactive >200”, “high monetary >1000”… phải được cấu hình và giải thích là business heuristic hoặc empirical quantile, không tự xưng data-driven nếu chưa đối chiếu dữ liệu.
- KPI đề xuất: incremental conversion, repeat rate, AOV, campaign cost, margin nếu có; không ghi lợi ích % hay ROI đạt được khi chưa có A/B test.
- Chọn biểu đồ làm rõ so sánh và insight; sửa filter KMeans/K-Means, palette giới hạn hai nhóm, clip axes làm ẩn score và nhãn tiền tệ sai.
- Không xuất PNG trắng chỉ để qua exists/size checks. Kiểm tra dữ liệu nguồn và render bằng mắt; missing mandatory figure phải FAIL hoặc có N/A có lý do theo hợp đồng.
- Mọi narrative động phải derive từ bảng nguồn, kể cả summary dashboard và executive summary. Không hard-code số cụm/tên nhóm/rate/top features/rules.
- Liên kết hình/report dùng relative path đúng từ vị trí notebook/report, không `file:///D:/...` hoặc link chỉ chạy trên máy tác giả.
- Giữ và hoàn thiện report Markdown + summary JSON + evidence manifest; báo cáo 07 phải có limitations và baseline discussion, không chỉ các bảng lớn.

### Output tối thiểu

Ba comparison tables, selection/artifact consistency audit, merge/cohort coverage audit, segment insights, action plan, rule insights, feature interpretations đúng model, bộ hình cần thiết, report07 Markdown/JSON và provenance hợp lệ.

## 11. Hoàn thiện kiến trúc, tái lập và dashboard

### 11.1 Một nguồn logic cho notebook và runner

- Module hóa logic stage trong `src/`; runners là entry points mỏng, notebooks gọi cùng hàm và thêm narrative/display.
- Bước 01–03 hiện có nhiều code chạy ở top level; tránh import runner gây chạy cả pipeline. Có `main()` guard và API stage rõ ràng.
- Hợp nhất path/config; root không phụ thuộc máy Windows cụ thể hoặc cwd tình cờ. Xử lý stdout reconfigure tương thích môi trường không hỗ trợ nếu giữ.
- Không tạo hai implementation khác nhau giữa notebook và runner cho cleaning, feature, chọn model hoặc chọn rules.
- Ghi rõ notebook cần output stage trước; chạy 01→07 từ dữ liệu gốc phải sinh đủ prerequisite. Không buộc mỗi notebook tự chạy lại toàn dự án.
- Stage 02/03/07 phải thực sự sinh output của mình qua shared functions; có thể tái sử dụng cache đã xác thực input/code/config hash, không âm thầm dùng PNG cũ.
- Rà code chết/module rỗng như `src/rfm_analysis.py`; hoặc hợp nhất/document rõ chức năng nằm ở đâu, không mô tả module rỗng như đã triển khai.
- Không nuốt mọi exception hoặc tắt toàn bộ warnings. Ghi convergence/deprecation warnings liên quan và cách xử lý; lỗi phải có ngữ cảnh stage/config.
- Tiền xử lý học từ dữ liệu và inference dùng chung pipeline/contract; thứ tự feature, encoding, threshold phải thống nhất.

### 11.2 Môi trường và dữ liệu

- Chốt Python và dependency versions đã kiểm thử; thêm lock/constraints hoặc tài liệu môi trường có thể tái lập. Không chỉ để tất cả `>=` rồi nói hoàn toàn reproducible.
- Kiểm tra các phụ thuộc đang dùng thực sự, gồm markdown table export/`tabulate`, notebook execution/export/`nbclient`, `nbconvert`, `nbformat` nếu cần.
- Không bịa version từ báo cáo cũ; ghi phiên bản bằng runtime.
- README phải mô tả đúng cây file tồn tại, lệnh cài, nguồn data, kiểm tra schema/hash, full run, tests, notebook, demo, runtime/tài nguyên đo được và cách troubleshoot.
- Nếu CSV chuyển đổi từ XLSX hoặc nguồn khác, ghi provenance/encoding/date parsing và chứng minh tính tương đương cần thiết; không giả định hai file cùng tên là cùng dataset.
- Raw data/models không nhất thiết phải commit. Cung cấp hướng dẫn lấy raw và tạo bộ artifact nộp bài có kiểm soát; không công khai dữ liệu/credentials hoặc auto-upload.
- Không tự chọn/thay giấy phép code nếu chưa có quyết định chủ sở hữu; sửa README đang trỏ LICENSE không tồn tại hoặc đánh dấu cần xác nhận. Ghi nguồn/attribution dữ liệu riêng.

### 11.3 Pipeline contract và manifest

- Có danh mục đầy đủ output bắt buộc/optional/conditional của từng stage, schema, unique keys, units, điều kiện empty-valid và validators.
- Pipeline giữ hành vi exit nonzero khi bước lỗi hoặc thiếu bắt buộc; mở rộng từ exists/size tới parse/schema/semantics.
- Bản ghi run có run_id, timestamps thực, commit HEAD lúc chạy, dirty-state/source digest khi có thay đổi chưa commit, environment versions, raw hash, config hash, seed, stage timings, input/output checksums và trạng thái validation.
- Không sửa trường commit thành một SHA khác chỉ để nhìn “mới”. Artifacts thường được tạo trước commit chứa chúng; cần giải thích provenance thay vì giả mạo thời gian.
- Không bỏ qua output missing trong manifest rồi đánh dấu toàn bộ PASS. Ghi từng failure, warning, skipped/N/A đúng nghĩa; overall status được tính từ các checks.
- Hash manifest sau khi hoàn tất qua một file/index ngoài nếu cần; tránh cơ chế manifest tự chứa hash của chính nó.
- Tách runtime/timestamp là trường không deterministic; so reproducibility bằng dữ liệu/metric có tolerance thích hợp, không yêu cầu mọi byte PNG hoặc pickle giống nhau trên mọi máy.
- Có chế độ chạy vào output root mới để nghiệm thu không dựa file cũ; không xóa đệ quy thư mục rộng hoặc raw/user outputs để giả lập clean run.

### 11.4 Dashboard minh họa thực sự

Nâng cấp app Streamlit hiện có vừa đủ minh họa ba bài toán, không làm ứng dụng production quá phạm vi:

1. Tổng quan dữ liệu/phạm vi/cleaning và run metadata.
2. Customer segmentation: tra cứu cụm và profile, giải thích K và giới hạn.
3. Repeat prediction: mẫu input đúng schema, upload CSV, validate, predict bằng artifact + threshold thực, hiển thị score, cảnh báo và tải kết quả.
4. Association rules: lọc/tra cứu sản phẩm, support counts/confidence/lift và giới hạn.
5. Model comparison & insights: baseline, selection rationale và hành động đề xuất.

Có thể tổ chức thành tabs/pages phù hợp; không cần đúng 5 tab nếu trải nghiệm tốt hơn với cấu trúc khác.

- Input lỗi/CSV rỗng/thiếu cột/type sai/Inf/âm không hợp lệ phải có thông báo rõ; missing feature được phép impute theo contract thì không bị cấm vô lý. Unknown categories được xử lý đúng Pipeline.
- App dùng đường dẫn theo project root, kiểm tra model/config version; không retrain khi mỗi widget đổi.
- Model chưa sẵn sàng triển khai hoặc chưa vượt baseline ở tiêu chí đích phải hiện cảnh báo phù hợp, không quảng cáo predictive accuracy.
- Chỉ load pickle/joblib do chính pipeline đáng tin cậy tạo; không cho upload model tùy ý rồi unpickle.
- Không giả vờ predict khách mới bằng DBSCAN/Agglomerative nếu chưa có phương pháp hợp lệ.
- Chạy smoke test và lưu minh chứng demo thật: ảnh chụp một số màn hình chính và kịch bản thao tác ngắn. Không tạo ảnh giả giao diện như bằng chứng app chạy.
- Không bắt buộc public hosting hoặc email/campaign thật. Demo local có thể đáp ứng mục tiêu minh họa.

## 12. Hoàn thiện báo cáo và tổng kết dự án

Tạo/cập nhật một báo cáo tổng kết theo CRISP-DM, nguồn Markdown có thể tái tạo, kèm bản PDF hoặc DOCX phù hợp yêu cầu nộp nếu môi trường có công cụ. Không báo có file PDF nếu chưa tạo/mở kiểm tra.

Nội dung chính:

- Tên đề tài, phạm vi, người dùng kết quả, executive summary.
- Business Understanding: business questions → data-mining tasks → metric/evidence → quyết định hỗ trợ; tiêu chí thành công kỹ thuật khác KPI business thực đạt.
- Data Understanding: nguồn, cấu trúc, chất lượng, EDA có số liệu, vấn đề và rủi ro.
- Data Preparation: cleaning, feature/label/basket definitions, point-in-time correctness, lý do và tác động downstream.
- Modeling: phương pháp, giả định, preprocessing, search spaces, protocol, budget và baseline.
- Evaluation: so sánh phản biện, bất định, chọn model/K/threshold, error analysis, rules equivalence, baseline deltas, giới hạn generalization.
- Deployment/Demo: kiến trúc minh họa, cách chạy, input/output và phạm vi không hỗ trợ.
- Insights/action plan với số liệu và thử nghiệm cần có để kiểm chứng hiệu quả.
- Hạn chế, các thất bại/không cải thiện đã quan sát, hướng phát triển có thứ tự ưu tiên.
- Tổng kết nhóm, tài liệu tham khảo, AI-use disclosure, phụ lục thực nghiệm và evidence index.

Yêu cầu:

- Số liệu thống nhất từ notebook → CSV → summary → report → app; bảng/hình có caption, đơn vị, cohort, tên metric và reference tới output.
- Đưa bảng quá rộng/toàn bộ search results vào phụ lục; thân báo cáo chọn thông tin cần để giải thích quyết định, không chỉ paste tất cả CSV.
- Cập nhật `business_understanding.md`, `methodology.md`, `data_dictionary.md`, các report01/02 và README để không giữ nội dung cũ mâu thuẫn.
- Không viết đã “cải thiện độ chính xác cao nhất” sau dropping missing ID khi chưa có phép đánh giá; không gọi mọi cột đã điền là phục hồi ground truth.
- `team_tasks.md` cần tên/việc thực tế, sản phẩm, commit/PR hoặc minh chứng và người xác nhận. Commit count không tự động bằng chất lượng đóng góp.
- Nếu chưa có thông tin nhóm/góp ý giảng viên, tạo danh sách trường cần bổ sung; không tự viết đã hoàn thiện góp ý hoặc tạo lịch sử họp.
- Chuẩn bị template peer assessment theo rubric, để thành viên tự đánh giá. Tôn trọng tính riêng tư; không commit công khai phiếu cá nhân bí mật.
- Ghi rõ công cụ AI và phần được hỗ trợ, những kiểm chứng do nhóm thực hiện; không giả vờ người thật đã review khi chưa có.
- Bảng rubric cuối cùng phản ánh việc còn thiếu do con người cung cấp. Không đánh dấu criterion 6 hoàn tất chỉ vì tạo một template.

## 13. Bộ kiểm thử bắt buộc

Bổ sung unit/regression tests nhỏ, deterministic, không cần raw data lớn; tách integration/end-to-end test dùng dữ liệu thật. Synthetic fixtures phải được nhận diện là test-only.

### 13.1 Dữ liệu và preprocessing

- CSV/XLSX loader/schema, date parse và encoding errors không bị catch che mất nguyên nhân.
- Missing ID không biến thành “nan”; invalid numeric không âm thầm thành số hợp lệ; raw hash không đổi.
- Flags có thể overlap nhưng removal accounting không đếm đôi.
- Duplicate policy trước/sau imputation; StockCode ngoại lệ; customer/product branches đúng parent.
- Missing Description không làm biến mất item khi tạo basket.
- Outlier policy/threshold không học từ validation/test khi dùng cho supervised model.

### 13.2 Features và leakage

- RFM tính tay, unique invoice frequency, Monetary/AOV và sample aggregation.
- R-score đúng hướng trong qcut fallback, tied values, all-equal, n<bins và thay thứ tự dòng.
- Cutoff boundary, label_end, khách mới sau cutoff, right-censoring, horizon khác 90.
- Thêm giao dịch tương lai không đổi features ở cutoff cố định; đối chiếu các aggregation quan trọng chứ không chỉ kiểm tên hai cột future.
- Future audit columns/CustomerID/target không vào X; feature schema và join indices ổn định.
- Basket dense/from-long tương đương, true presence không dựa Description.

### 13.3 Modeling

- Clustering one-cluster/all-noise được đánh dấu N/A đúng; config_id/selected row/model artifact nhất quán.
- Selection filter/rank/tie-break được kiểm chứng, không fallback che mismatch.
- Train/test và folds không giao CustomerID theo protocol; transforms chỉ fit trong phạm vi cho phép.
- Tuning/threshold selection không truy cập holdout; selection không đổi khi chỉ tráo test metrics trong fixture.
- Case Dummy vượt learned candidate vẫn được báo trung thực; không viết test chỉ yêu cầu “winner phải khác Dummy” như bằng chứng chất lượng.
- Feature schema/unknown category/permitted missing/invalid inputs; prediction ID alignment.
- Save/load parity cho score và label dưới threshold được lưu.
- Case pipeline file bị hỏng, estimator class/metadata khác, threshold khác phải bị phát hiện.

### 13.4 Rules và insights

- Rule keys bằng StockCode; description chứa dấu phẩy không phá overlap check.
- Hai bộ rules cùng số lượng nhưng khác nội dung phải FAIL equality.
- Numerical tolerance, out-of-range metrics, valid conviction +∞ và undefined conviction.
- No valid rules → empty schema + explicit status, không fake success/error ambiguity.
- Algorithm selection khớp rules thực được chọn.
- Segment joins có duplicate/missing IDs, cohort coverage và mẫu số khác nhau; không diễn giải full-history retrospective rate như predictive validation.
- `selected` alias và string boolean parse đúng; không bypass artifact verification.

### 13.5 Pipeline/notebook/report/demo

- Thiếu output bắt buộc, file 0-byte, CSV header-only trái contract, schema sai, stale run/model/PNG: phải bị phát hiện.
- Plot filter không tìm thấy dữ liệu phải báo lỗi có ngữ cảnh hoặc N/A hợp lệ; không tạo PNG trắng.
- Không đọc nguồn trước có “PASS” rồi giả định hiện tại cũng PASS; verify manifest semantics.
- App smoke test: healthy load, empty input, thiếu cột, unknown category và threshold parity.
- Report numbers được đối chiếu machine-readable với metrics source; link paths tới files đúng.

Không có yêu cầu ngụy tạo mọi metric đều finite: phân biệt metric undefined hợp lệ, failure thực sự và null do thiếu dữ kiện. Các biến/đầu ra có quy định không thiếu thì enforce chặt.

## 14. Thực nghiệm tối thiểu và ngân sách

Lập bảng này trước khi chạy, điền cấu hình thực và artifact sau khi chạy:

| Experiment | Câu hỏi | Đối chứng công bằng | Minh chứng |
|---|---|---|---|
| E01 | Cleaning làm thay đổi dữ liệu gì? | Raw audit → các bước lọc có parent rõ | Counts/IDs/value ledger, retained population, missing bias |
| E02 | Log/scale/robust handling có ích không? | Cùng dữ liệu hợp lệ, thay một policy; kiểm soát geometry/cohort | Data distribution, downstream metrics và stability |
| E03 | Có cần biến ngoài RFM? | RFM vs thêm behavior vs Country trên cùng folds | CV metrics, dispersion, runtime, decision |
| E04 | Vì sao algorithm/K được chọn? | K grid và thuật toán cùng representation | Internal metrics, invalid reasons, combined rank, profiles |
| E05 | Lựa chọn cụm ổn định đến đâu? | Finalists trên cùng khách qua nhiều seed | Metric distribution và partition ARI, limitation |
| E06 | Learned models hơn baseline ở đâu? | Dummy + LR/DT/RF cùng train/CV/test protocol | Primary/secondary scores và delta baseline |
| E07 | Tuning có ích không? | Default vs tuned với selection/evaluation tách rõ | Search results, params, valid performance comparison |
| E08 | Threshold ảnh hưởng FP/FN thế nào? | Ngưỡng mặc định vs chọn từ train validation/OOF | Curves, confusion counts, constraint/utility assumptions |
| E09 | Apriori/FP-Growth có cùng kết quả không? | Cùng basket/config, canonical content comparison | Set differences, metric deltas, repeated runtimes |
| E10 | Support/confidence ảnh hưởng luật gì? | Lưới nhỏ có kiểm soát | Counts/coverage/quality/cost, chosen thresholds |
| E11 | Insight liên kết có đúng population/time không? | Full cohort vs label-eligible cohort được phân biệt | Join coverage, numerator/denominator, time audit |

Không cần mọi tổ hợp của mọi thí nghiệm. Dùng budget hợp lý, tái sử dụng computed results bằng cache có checksum, ưu tiên đáp ứng rubric. Nếu giảm budget, ghi giảm gì và hạn chế, không tự lược bỏ tuning/ablation rồi gọi là đầy đủ.

## 15. Trình tự triển khai và chạy nghiệm thu

### Giai đoạn A — Audit và hợp đồng

1. Đọc toàn project; ghi commit/dirty-state.
2. Lập audit checklist từ A01–A19 và phát hiện bổ sung.
3. Chốt output contract, experiment protocol, file naming, data/feature/model/rule schemas.
4. Chốt tiêu chí selection trước khi có kết quả mới; ghi hạn chế holdout đã từng được xem.

### Giai đoạn B — Sửa logic và regression tests

5. Sửa 01–03 và tests dữ liệu/feature trước.
6. Sửa 04–06: selection, tuning, ablation, rules comparison, artifacts.
7. Sửa 07: provenance, cohort/time, interpretation và dynamic report.
8. Đồng bộ notebooks/runners/shared functions/app/docs; không làm hai nhánh logic riêng.

### Giai đoạn C — Chạy thật và diễn giải

9. Chạy unit tests, sửa mọi failure trong phạm vi.
10. Chạy pipeline đầy đủ trên raw data thật vào output root sạch/mới.
11. Chạy các thí nghiệm đã chốt; ghi cả cấu hình thất bại/không cải thiện.
12. Viết narrative dựa kết quả thực; liên kết evidence_id tới bảng/hình.
13. Chạy toàn bộ 7 notebook từ kernel mới, thực thi cells đúng thứ tự, không `allow_errors=True`, không gán execution_count hoặc chèn output bằng tay.
14. So runner và notebook: data/selection/metrics phù hợp tolerance, cùng semantic config; runtime/timestamp có thể khác. Nếu run notebook tạo lượt mới, cập nhật provenance đúng lượt đó, không trộn nguồn.
15. Export HTML notebook để đọc không cần kernel; output inline phải hiện bảng/biểu đồ chính. Markdown tĩnh và execution output đều cần đọc mạch lạc.
16. Chạy app thực, lưu demo evidence và kịch bản.
17. Tạo final report từ nguồn kết quả, kiểm tra layout/render khi xuất PDF/DOCX, kiểm tra các đường dẫn.

Các lệnh gốc cần giữ hoạt động, điều chỉnh/document khi bổ sung flags:

```bash
python -m compileall -q src notebooks app tests run_pipeline.py
python -m pytest -q
python run_pipeline.py
python -m streamlit run app/app.py
```

Tạo một entry point rõ ràng để execute/validate toàn bộ notebook và export HTML, ví dụ script trong `scripts/`; chỉ ghi lệnh hoàn chỉnh sau khi đã triển khai và chạy. Không đưa vào README một lệnh tưởng tượng.

### Giai đoạn D — Nghiệm thu cuối

18. Chạy validator toàn dự án: schemas, output inventory, provenance, model parity, rules equivalence, notebook errors/execution, report references.
19. Kiểm tra bằng mắt các hình quan trọng: không trắng, không nhãn đè đến mức không đọc, không sai đơn vị, không che dữ liệu bất lợi.
20. Điền rubric matrix bằng minh chứng thật; liệt kê `NEEDS_USER_INPUT`/BLOCKED riêng.
21. Bàn giao bản tóm tắt thay đổi và cách kiểm tra lại; không tự push.

## 16. Bộ sản phẩm phải bàn giao

Có thể chọn tên mới hợp lý, nhưng cần một artifact index chỉ rõ tên/path thực; không tạo nhiều tài liệu rời trùng nội dung:

- 7 notebook hoàn chỉnh, executed, không traceback; output có thể đọc được.
- 7 runner nhất quán với shared modules và pipeline đầu–cuối.
- Unit/regression/integration tests, execution logs và validation summary.
- Datasets/intermediate outputs, model artifacts, tables/figures đủ theo contract; có cách tái tạo nếu không commit.
- Experiment registry + full search/fold results + ablation/sensitivity + selection audits.
- Feature decision dictionary, decision log, methodology, references, rubric evidence matrix.
- Dashboard chạy được, sample input hợp lệ, kịch bản demo và bằng chứng thật.
- Report 07; final report CRISP-DM; cập nhật README/docs.
- Run manifest/provenance và `outputs/evidence/artifact_index.md` hoặc tương đương.
- Danh sách dữ kiện cần nhóm bổ sung: tên/đóng góp/peer assessment/góp ý tiến độ/AI disclosure xác nhận.

## 17. Definition of Done — không được bỏ qua

Chỉ kết luận “hoàn thiện phần kỹ thuật” nếu tất cả điều kiện bắt buộc dưới đây đã được kiểm chứng:

- [ ] Code hiện tại chạy được với môi trường đã ghi lại.
- [ ] Pipeline tạo đủ output từ raw trong một lần nghiệm thu không dựa stale files.
- [ ] 7 notebook chạy clean kernel, đúng thứ tự, không error output, có bảng/hình và diễn giải.
- [ ] Mỗi quyết định quan trọng có purpose/rationale/source hoặc empirical evidence/alternative/result/limitation.
- [ ] Feature/label thời gian đúng, có tests phát hiện leakage chứ không chỉ tuyên bố.
- [ ] Đã so preprocessing/features, tuning và baseline bằng protocol công bằng.
- [ ] K/model/threshold/association selection xuất phát từ bảng thực nghiệm, không hard-code.
- [ ] Lý do K=2 nếu được chọn được giải thích bằng metric, đối chứng, quy mô cụm, stability và hạn chế.
- [ ] Dummy không bị che giấu; kết quả learned model chưa tốt được báo trung thực.
- [ ] Rules hai thuật toán đã đối chiếu nội dung thực; conviction xử lý đúng.
- [ ] Model/rules đã lưu đúng lựa chọn; inference threshold và predictions khớp metadata.
- [ ] Insight không nhầm full-history với forecast; mọi rate có tử số/mẫu số/cohort.
- [ ] Report/chart/app không dùng số liệu kết quả hard-code hoặc đơn vị sai.
- [ ] Tests và validation có logs, failure có thể phát hiện bằng test âm.
- [ ] App đã smoke-test và có minh chứng demo thật.
- [ ] Tài liệu/README phản ánh đúng chức năng/file/lệnh đang tồn tại.
- [ ] Rubric matrix dẫn được tới bằng chứng đã kiểm tra.

“Hoàn thiện toàn bộ hồ sơ nộp” là trạng thái riêng: còn yêu cầu báo cáo cuối, thông tin nhóm thật, peer assessment và các thông tin bắt buộc khác của rubric. Nếu thiếu những dữ kiện này, ghi rõ **kỹ thuật hoàn tất / hồ sơ còn thiếu**, không tuyên bố hoàn tất toàn bộ.

Kết quả model thấp hơn kỳ vọng, K khác 2, không có rule hữu ích, hoặc chưa đủ temporal evidence không tự động là thất bại kỹ thuật. Báo cáo trung thực có phương pháp đúng tốt hơn kết quả đẹp nhưng sai. Chỉ đánh dấu scientific conclusion phù hợp, không bịa cải thiện để đạt checklist.

## 18. Phản hồi cuối cùng tôi muốn nhận từ bạn

Sau khi thực hiện, trả lời ngắn gọn nhưng có đường dẫn và minh chứng:

1. Trạng thái thực: kỹ thuật hoàn tất/chưa, hồ sơ nộp hoàn tất/chưa; không bảo đảm điểm số.
2. Bảng notebook 01–07: lỗi/thiếu ban đầu → thay đổi → output mới → test/log xác nhận.
3. Các quyết định cuối cùng: preprocessing/features, algorithm/K, classifier/threshold, association algorithm/thresholds và số liệu đối chứng chủ chốt.
4. Những kết quả không cải thiện hoặc còn hạn chế; baseline nào chưa vượt, việc gì chưa chứng minh.
5. Lệnh đã chạy, số tests passed/failed/skipped, dữ liệu/config/run_id và vị trí logs.
6. Vị trí notebook, báo cáo, bảng thực nghiệm, dashboard, demo evidence và artifact index.
7. Ma trận rubric 6 tiêu chí với trạng thái có minh chứng; còn thiếu gì do người dùng cung cấp.
8. Danh sách file thay đổi và cách chạy lại. Không chỉ nói “đã hoàn thiện”.

Hãy bắt đầu bằng audit hiện trạng thật, sau đó triển khai đến khi đạt tiêu chí nghiệm thu hoặc gặp blocker cần người dùng. Không dừng sau việc viết nhận xét và không chỉ thêm Markdown để che những thiếu sót trong code/phương pháp.