import { StyleSheet } from "react-native";

const TEXT_SOFT_DARK = "#101426";

export const styles = StyleSheet.create({
  safe: {
    flex: 1,
    backgroundColor: "#F7F8FB",
  },
  container: {
    padding: 20,
    paddingBottom: 40,
    backgroundColor: "#F7F8FB",
  },
  header: {
    paddingTop: 14,
    paddingBottom: 10,
  },
  headerTitle: {
    color: TEXT_SOFT_DARK,
    fontSize: 28,
    fontFamily: "Exo2-Bold",
  },
  headerLine: {
    flex: 1,
    height: 1,
    backgroundColor: "#E2E7F1",
    marginHorizontal: 12,
  },
  headerDate: {
    color: TEXT_SOFT_DARK,
    fontSize: 12,
    letterSpacing: 1,
    fontFamily: "Exo2-Regular",
  },
  dateBreakRow: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 12,
    marginTop: 4,
  },
  dateBreakLine: {
    flex: 1,
    height: 1,
    backgroundColor: "#E2E7F1",
    marginRight: 10,
  },
  dateBreakText: {
    color: TEXT_SOFT_DARK,
    fontSize: 11,
    letterSpacing: 0.8,
    textTransform: "uppercase",
    fontFamily: "Exo2-Regular",
  },
  row: {
    flexDirection: "row",
    alignItems: "flex-start",
    marginBottom: 0,
    paddingBottom: 10,
  },
  treeWrap: {
    width: 34,
    alignItems: "center",
    position: "relative",
    alignSelf: "stretch",
  },
  rail: {
    position: "absolute",
    top: -12,
    bottom: -12,
    width: 2,
    backgroundColor: "#E2E7F1",
  },
  nodeWrap: {
    position: "absolute",
    top: 18,
    left: 11,
    flexDirection: "row",
    alignItems: "center",
  },
  dot: {
    width: 12,
    height: 12,
    borderRadius: 6,
  },
  branch: {
    width: 12,
    height: 2,
    backgroundColor: "#E2E7F1",
  },
  card: {
    flex: 1,
    backgroundColor: "#FFFFFF",
    borderRadius: 16,
    padding: 16,
    borderWidth: 1,
    borderColor: "#E2E7F1",
  },
  timeText: {
    color: TEXT_SOFT_DARK,
    fontSize: 12,
    marginBottom: 6,
    fontFamily: "Exo2-Regular",
  },
  titleText: {
    color: TEXT_SOFT_DARK,
    fontSize: 16,
    fontFamily: "Exo2-Bold",
    marginBottom: 6,
  },
  bodyText: {
    color: TEXT_SOFT_DARK,
    fontSize: 13,
    marginBottom: 6,
    fontFamily: "Exo2-Regular",
  },
  metaText: {
    color: TEXT_SOFT_DARK,
    fontSize: 11,
    fontFamily: "Exo2-Regular",
  },
});
