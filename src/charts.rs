//! Terminal chart rendering utilities.

/// Format and print a two-column comparison table.
#[allow(dead_code)]
pub fn comparison_table(
    title: &str,
    col1_label: &str,
    col2_label: &str,
    data: Vec<(String, f64, f64)>,
) {
    println!("\n{title}");
    println!("{:<12}  {:>12}  {:>12}", col1_label, col2_label, "Ratio");
    println!("{}", "-".repeat(50));

    for (label, val1, val2) in data {
        let ratio = if val1 > 0.0 { val2 / val1 } else { 0.0 };
        println!(
            "{:<12}  {:>12.4}  {:>12.4}  ({:.2}x)",
            label, val1, val2, ratio
        );
    }
}
